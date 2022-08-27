"""
Module for defining the encoder sublayer. This will eventually wrap into the full SentenceEncoder
class
"""

import logging
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
from annotated_mpnet.transformer_modules import RelativeMultiHeadAttention


class SentenceEncoderLayer(nn.Module):
    """
    Implements the the layers within the full SentenceEncoder
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        normalize_before: bool = True,
        export: bool = False,
    ) -> None:
        """
        Init function for the layer. I will try to summarize all the args here.

        Args:
            embedding_dim: the embedding dimension for the layer. Should always be 768 really
            ffn_embedding_dim: this is the size of the hidden layer in the fully-connected
                subsection of the layer right AFTER self-attention. Also known as the feed-forward
                network component of the encoder layer
            num_attention_heads: the number of attention heads for each layer. Default here is 8,
                but you will usually want to bump this higher if your model is accepting longer
                sequence lengths
            dropout: pretty straightforward, but this is the dropout prob for the FC layers in the
                forward pass
            attention_dropout: similar to above, but is the dropout prob within the self-attention
                mechanism
            activation_fn: the activation function you will be using in this network. Although ReLU
                is the default, more and more evidence points towards GELU being better for large
                NLP-based transformers
            add_bias_kv: boolean that dictates whether or not to add a bias parameter to the K, V
                matrices in the self-attention mechanism
            add_zero_attn: boolean that dictate whether or not to add zero attention to the
                self-attention mechanism
            normalize_before: boolean that determines if the LayerNorm will be applied BEFORE or
                AFTER the self-attention calculation. Functionally they are very similar, but the
                standard is to normalize before
            export: boolean that would enable ONNX tracing for exporting, but I think we won't be
                using this
        """
        super().__init__()

        # Store args
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Get the submodules we need
        self.activation_fn = utils.get_activation_fn(activation_fn)

        # Initialize the self attention module
        self.self_attn = RelativeMultiHeadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
        )

        # Get the LayerNorm for the self_attention output
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        # Get the FC linear layers for the hidden connections in each layer after the self-attention
        # is calculated
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # Get the final LayerNorm
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        positions_bias: torch.Tensor = None,
    ):
        """
        Forward pass of the encoder layer. Calculates self-attention, does normalization (before or
        after depending on args), processes the skip connection, and uses the linear layers at the
        end
        """

        # Keep the residual for the skip connection after self-attention calculation
        residual = x

        # This is a bit of an overloaded function that will check if normalization should be
        # processed before or after self-attention. It will cross reference the "before" or "after"
        # kwarg against self.normalize_before arg and then either do nothing or normalize
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)

        # Forward pass of self-attention
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
            positions_bias=positions_bias,
        )

        # The below operations may look scary, but we will do our best to summarize their use

        # Process the dropout after self-attention is calcualted and then make the skip connection
        # by adding residual to x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # Try the maybe_layer_norm function again to potentially normalize after self-attention
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        # Now we must process the fully connected layer after self-attention. Similarly, there is
        # also a LayerNorm that must be calculated before or after (and is determined by the
        # normalize_before arg)

        # Save the residual for the skip connection after FC layer
        residual = x

        # Process the layer norm
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)

        # Process the first layer of the feed-forward network which expands the embedding from
        # embedding_dim to ffn_embedding_dim (Linear + activation)
        x = self.activation_fn(self.fc1(x))

        # Process the dropout once again
        x = F.dropout(x, p=self.activation_dropout, training=self.training)

        # Calculate the second portion of the feed-forward net, converting the hidden size back to
        # our embedding size of embedding_dim. This time we DO NOT add the activation function so
        # as to not kill the neurons
        x = self.fc2(x)

        # Process the droput again
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Calculate the skip connection with the residual and the output of the feed-forward net
        x = x + residual

        # Finally, process the LayerNorm once again
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        return x, attn

    def maybe_layer_norm(
        self, layer_norm: nn.Module, x: torch.Tensor, before=False, after=False
    ) -> torch.Tensor:
        """
        The key helper function that will only trigger if the before/after bool passed in matches
        what is dictated by the self.normalize_before
        """
        # First make sure before and after both aren't true with a quick XOR
        assert before ^ after, "You must set only one of 'before' or 'after'"

        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
