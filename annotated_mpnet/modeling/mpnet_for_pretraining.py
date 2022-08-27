"""
Module containing the necessary classes for MPNet pretraining, ported directly from fairseq research
code
"""

import logging
from typing import Tuple
from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)

import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

from annotated_mpnet.utils import utils
from annotated_mpnet.transformer_modules import LayerNorm
from annotated_mpnet.transformer_modules import SentenceEncoder


def init_final_params(module: nn.Module) -> None:
    """
    This is a function that does the final initialization of weights as according to the original
    BERT paper. This is very important to make sure all biases are zeroed out to start and that
    embedding and linear layers start within a normal distribution at instantiation.

    Args:
        module: this is a module within the model. We will use nn.Module's builtin `apply` function
            to apply this as a callable to all submodules
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx:
            module.weight.data[module.padding_idx].zero_()


class MPNetForPretraining(nn.Module):
    """
    Class containing all the methods required for pretraining MPNet
    """

    def __init__(self, args, tokenizer) -> None:
        super().__init__()

        # Let's define the encoder here
        self.args = args
        self.sentence_encoder = SentenceEncoder(
            padding_idx=tokenizer.vocab[tokenizer.pad_token],
            vocab_size=tokenizer.vocab_size,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            activation_fn=args.activation_fn,
            normalize_before=args.normalize_before,
        )

        # Add the language modeling head so that we can do pretraining
        self.lm_head = MPNetLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=tokenizer.vocab_size,
            activation_fn=args.activation_fn,
            weight=self.sentence_encoder.embed_tokens.weight,
        )

        # Finally initialize the weights according to the guidelines in the original BERT paper
        self.apply(init_final_params)

    def output_layer(
        self, features: torch.Tensor, masked_tokens: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Wrapper function for language modeling output layer
        """
        return self.lm_head(features, masked_tokens)

    def forward(self, input_ids, positions, pred_size, return_mlm=False, **kwargs) -> torch.Tensor:
        """
        Forward function for computing MPNet
        """

        # Calculate initial embeddings
        emb = self.encode_emb(self.sentence_encoder, input_ids, positions)

        # Reverse the tensor for easier extraction
        x = reverse_tensor(emb)

        # Separate out content and query streams
        c, q = split_tensor(x, pred_size)

        # Get the content and query position biases
        content_position_bias = self.encode_relative_emb(
            self.sentence_encoder, positions[:, :-pred_size]
        )
        query_position_bias = content_position_bias[:, -pred_size:].contiguous()

        # Get the sz of the inital src_length without the tokens to be predicted
        sz = c.size(0) - pred_size

        # Get the query and content masks using the helper function below
        query_mask, content_mask = make_query_and_content_mask(input_ids, sz, pred_size)

        # Do the attention calculations
        for i, layer in enumerate(self.sentence_encoder.layers):
            c, q = encode_two_stream_attention(
                layer, c, q, content_mask, query_mask, content_position_bias, query_position_bias
            )

        # Process the final layer norm
        q = self.maybe_final_norm(self.sentence_encoder, q)

        # Re-reverse the tensor so we can have it back in the correct format
        q = reverse_tensor(q)

        # Project the attention features out to the vocab size for masked token classification
        x = self.output_layer(q)

        # If we also want MLM loss, we can branch to the below logic. Probably not useful for us
        if return_mlm is True:
            c = c[-pred_size:]
            c = self.maybe_final_norm(self.decoder.sentence_encoder, c)
            c = reverse_tensor(c)
            c = self.output_layer(c)
            return x, c

        return x

    # We define some class static methods here that will be used quite a bit across the board
    @staticmethod
    def encode_emb(self, input_ids: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        """
        Method for embedding the input tokens (i.e. input_ids)

        Args:
            input_ids: the input IDs for the given batch
            positions: the position values
        """

        # Use the embedding layer of the sentence encoder to embed these (passed in via the self
        # arg)
        x = self.embed_tokens(input_ids)

        # Scale the embeddings if necessary
        if self.embed_scale is not None:
            x *= self.embed_scale

        # Add in positions
        if positions is not None:
            x += F.embedding(positions + 2, self.embed_positions.weight, self.padding_idx)

        # Do layer norm
        if self.emb_layer_norm is not None and not self.normalize_before:
            x = self.emb_layer_norm(x)

        # Process dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    @staticmethod
    def maybe_final_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Another helper function to process the final layer norm if necessary
        """
        if self.emb_layer_norm is not None and self.normalize_before:
            return self.emb_layer_norm(x)
        return x

    @staticmethod
    def encode_relative_emb(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Helper function to properly handle relative position bias
        """
        qlen, klen = positions.size(1), positions.size(1)
        context_position = positions[:, :, None]
        memory_position = positions[:, None, :]

        relative_position = memory_position - context_position

        rp_bucket = self.relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(positions.device)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute(0, 3, 1, 2).contiguous()  # [bsz, head, qlen, klen]
        values = values.view(-1, qlen, klen)
        return values


class MPNetLMHead(nn.Module):
    """
    Head for language modeling on the output of MPNet
    """

    def __init__(self, embed_dim: int, output_dim: int, activation_fn: str, weight=None) -> None:
        """
        Let's talk about these args so we can better understand what's happening in the LM head

        Args:
            embed_dim: the embedding dimension of the encoder model (usually 768)
            output_dim: the dimension that we want to project out to (usually the vocab size)
            activation_fn: the activation to be using within the LM head
        """
        super().__init__()

        # Let's define the layers for the LM head. It's a pretty simple pipeline

        # Dense FC layer before casting the embed to the vocab size
        self.dense = nn.Linear(embed_dim, embed_dim)

        # Activation function
        self.activation_fn = utils.get_activation_fn(activation_fn)

        # Get the layer norm
        self.layer_norm = LayerNorm(embed_dim)

        # If we don't provide our own weights, we need to initialize them
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight

        # Finally create the bias layer
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None):
        """
        Forward pass for the LM head

        Args:
            features: outputs of the encoder portion
            masked_tokens: which tokens are masked
        """

        # Only project the unmasked tokens while training, saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        # Step through the network
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)

        # Project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias

        return x


# Helper functions below!
def reverse_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    This function simply reverses a tensor. This will be used to make extracting content and query
    streams easier later on
    """
    return x.transpose(0, 1)


def split_tensor(x: torch.Tensor, split_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This helper function separates the query stream from the content stream in the input IDs

    Args:
        x: the tensor to split
        split_size: the pred_size of the input ID sequence
    """
    # Get the content stream size by subtracting out the pred_size aka split_size
    sz = x.size(0) - split_size

    return x[:sz].contiguous(), x[sz:].contiguous()


def encode_two_stream_attention(
    self,
    c: torch.Tensor,
    q: torch.Tensor,
    content_mask: torch.Tensor = None,
    query_mask: torch.Tensor = None,
    content_position_bias: torch.Tensor = None,
    query_position_bias: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This helper function wraps the two-stream attention calculation and calculates the skip
    connection as well as layer norm. This function is actually used by passing in a class instance
    under the self arg for each LAYER in the encoder

    Args:
        c: the content portion of the input sequence (can be thought of as containing the
            appropriate information to encode the bidirectional nature of the encoder attention, but
            is kept separately otherwise the query attention mechanism would always be able to
            predict its own content trivially)
        q: the query portion of the input sequence (can be thought of as the <mask> tokens we
            are looking to predict using language modeling)
        content_mask: the attention mask for the content stream
        query_mask: the attention mask for the query stream
        content_position_bias: position bias for the content portion
        query_position_bias: position bias for the query portion

    Returns:
        A tuple containing the content and query tensors after the attention calculation is done for
        the given layer in self
    """

    def skip_norm_ff_fn(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Inner function that process all the normalization, skip connection, and the feed-forward net
        after the attention calculation. Since we do this for c and q, it's easier to keep this as
        a reusable function
        """

        # Calculate dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Process skip connection
        x = x + residual

        # Do normalization where appropriate based on the normalize_before param
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        # Save x as residual for the skip connection AFTER the feed-forard net
        residual = x

        # Normalize again
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)

        # Process the feed-forward net connections with specified activation function and dropout
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Process the skip connection after running through the FF net
        x = x + residual

        # Do a layer norm based on args
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        return x

    # Save c and q as residuals for skip connection after attention calculation
    residual_c = c
    residual_q = q

    # Do a normalization before if the class args allow for it
    c = self.maybe_layer_norm(self.self_attn_layer_norm, c, before=True)
    q = self.maybe_layer_norm(self.self_attn_layer_norm, q, before=True)

    # Wrapper function on top of each layer's self attention mechanism that calculates the proper
    # two stream attention that is required for MPNet
    c, q = two_stream_self_attention(
        self.self_attn,
        query=[c, q],
        key=c,
        value=c,
        query_mask=query_mask,
        content_mask=content_mask,
        query_position_bias=query_position_bias,
        content_position_bias=content_position_bias,
    )

    # Calculate skip connection, inner layer norms, and feed forward after attention calculation
    # using the resuable function we built above
    c = skip_norm_ff_fn(c, residual_c)
    q = skip_norm_ff_fn(q, residual_q)

    # Finally return the tensors after the full layer calculation
    return c, q


def two_stream_self_attention(
    self,
    query: torch.Tensor,
    key: torch.Tensor = None,
    value: torch.Tensor = None,
    query_mask: torch.Tensor = None,
    content_mask: torch.Tensor = None,
    query_position_bias: torch.Tensor = None,
    content_position_bias: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function is a wrapper on top of the encoder self attention (which is passed in using the
    `self` class instance keyword) that properly calculates the two stream self attention that is
    required for MPNet.

    Args:
        query: the tensors represening the Q matrix in self attention. We acttually pass in two here
            because of two stream attention
        key: the K matrix of the attention calculation
        value: the V matrix of the attention calculation
        query_mask: the attention mask for the query stream
        content_mask: the attention mask for the content stream
        query_position_bias: position bias for the query portion
        content_position_bias: position bias for the content portion
    """

    # Unpack the content and query tensors from the (poorly) named query arg
    c, q = query

    # Get dimensions
    bsz, embed_dim = key.size(1), key.size(2)

    # Define a few in-scope helper functions that we will be reusing a bunch
    def transpose_fn(x: torch.Tensor) -> torch.Tensor:
        """
        A reusable transpose function that matches the appropriate shape for this attention
        calculation (matching the head dimension and the number of attention heads)
        """
        return x.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def fill_mask(attn_weights: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Helper function that will apply the attention mask to the tensor containing the weights.
        This is done using the builtin `masked_fill` function, but we need to apply some additional
        processing on top
        """
        return attn_weights.masked_fill(attn_mask.unsqueeze(0), float("-inf"))

    def attn_fn(
        _q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
        bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        This is the adjusted attention function that still uses the core weights of the encoder, but
        processes everything differently to fit the two stream attention scheme
        """
        # Process the query matrix through both the scaling and the input layer of self_attention
        _q = transpose_fn(self.scaling * self.in_proj_q(_q))

        # Calculate the energy by multiplying Q and K
        attn_weights = torch.bmm(_q, k.transpose(1, 2))

        # Process bias if applicable
        if bias is not None:
            attn_weights += bias

        # Process attention masking
        if mask is not None:
            attn_weights = fill_mask(attn_weights, mask)

        # Softmax the energy to get the final attention weights
        attn_weights = F.softmax(attn_weights, dim=-1).type_as(attn_weights)

        # Do the attention dropout
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Combine the final post-softmax/dropout weights with the V matrix to get the attention
        attn = torch.bmm(attn_weights, v)

        # Finally, transpose back to the embed dimension and return
        attn = attn.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)

        return self.out_proj(attn)

    # Get K and V matrices by processing them through the input layer for each matrix and transpose
    # them to be the right shape
    k = transpose_fn(self.in_proj_k(key))
    v = transpose_fn(self.in_proj_v(value))

    # Calculate query attention and content attention using the function above
    c = attn_fn(c, k, v, mask=content_mask, bias=content_position_bias)
    q = attn_fn(q, k, v, mask=query_mask, bias=query_position_bias)

    return c, q


def make_query_and_content_mask(
    input_ids: torch.Tensor, seq_len: int, pred_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function makes the rather unique content and query mask that is required for MPNet.

    Args:
        input_ids: the batch of input IDs that the forward pass takes in. It is only used to send
            the mask to the right device, so it's a little useless. A refactor would just pass the
            device name in
        seq_len: the sequence length of the input
        pred_size: the size of the subsequence that is being converted to mask/corrupt tokens

    It looks like the below with comparisons to how it's different than XLNet-style PLM:
        Query Mask:
        | <-   PLM   -> |    | <-      MPNet    -> |
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 0 1 1 ]    [ 0 0 0 0 0 1 1 1 0 0 ]
        [ 0 0 0 0 0 0 1 ]    [ 0 0 0 0 0 0 1 1 1 0 ]
        Content Mask:
        | <-   PLM   -> |    | <-      MPNet    -> |
                               x x x x x x x m m m
                               1 2 3 4 5 6 7 5 6 7
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 0 1 1 ]    [ 0 0 0 0 0 1 1 1 0 0 ]
        [ 0 0 0 0 0 0 1 ]    [ 0 0 0 0 0 0 1 1 1 0 ]
        [ 0 0 0 0 0 0 0 ]    [ 0 0 0 0 0 0 0 1 1 1 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]
    """

    # Define helper function to keep things organized
    def make_query_mask():
        # Create the mask portion (i.e. ones)
        mask = torch.triu(torch.ones(pred_size, pred_size), 0)

        mask = (torch.ones(pred_size, seq_len - pred_size), 1 - mask, mask)

        return torch.cat(mask, dim=-1).eq(0)

    def make_content_mask():
        mask = [
            torch.zeros(seq_len - pred_size, pred_size),
            torch.tril(torch.ones(pred_size, pred_size), 0),
        ]

        mask.append(torch.zeros(pred_size, pred_size))
        mask = torch.cat(mask, dim=0)
        mask = (torch.ones(seq_len + pred_size, seq_len - pred_size), mask, 1 - mask)

        return torch.cat(mask, dim=-1).eq(0)

    return make_query_mask().to(input_ids.device), make_content_mask().to(input_ids.device)
