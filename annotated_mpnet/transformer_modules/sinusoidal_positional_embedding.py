"""
Module containing the SinusoidalPositionalEmbedding option, which creates a sinusoidal relationship
between a token and its position
"""

import logging
from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)

import os
import sys

import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.onnx.operators

from annotated_mpnet.utils import utils


class SinusoidalPositionalEmbedding(nn.Module):
    """
    A module for creating positional embeddings that follow a sinusoidal relationship
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024) -> None:
        super().__init__()

        # Store args
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Get the weights from a helper function that processes the sinusoidal math
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

        # Set the ONNX trace variable again, but I don't think we'll be using it
        self.onnx_trace = False

        # This is a builtin nn.Module function for registering buffer values within a module
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    def prepare_for_onnx_export(self):
        """
        This function is probably useless for us, but we keep it in
        """
        self.onnx_trace = True

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """
        This static method instantiates the sinusoidal pattern for the positional embeddings
        """

        # First get half the embedding dimension size
        half_dim = embedding_dim // 2

        # Not quite sure of the math happening below, but generally, we are constructing initial
        # weights using an algorithm that heavily involves trigonometric relationships (as you can
        # see in the last step with sin() and cos() making an appearance)
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)

        # Next calculate padding. If embedding size is not divisible by 2, we need to pad out
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)

        # If there IS a padding index, reset the weights to 0
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, **kwargs):
        """
        The forward function for processing these positional embeddings. Input should be of size
        (bsz x seq_len) as usual
        """

        # Break out dimensions
        bsz, seq_len = torch.onnx.operators.shape_as_tensor(input)

        # Get the max position of the given sequence
        max_pos = (self.padding_idx + 1) + seq_len

        # Now we add the option to recompute embeddings if the initial embeddings weren't large
        # enough to cover all positons
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )

        self.weights = self.weights.to(self._float_tensor)

        # Process incremental state below
        # Again, not really sure what this does
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        # Use the typical `make_positions` util to get incremental positions. This will eventually
        # feed directly into the sinusoidal weights we defined before
        positions = utils.make_positions(input, self.padding_idx, onnx_trace=self.onnx_trace)

        # If onnx_trace is set (which it shouldn't be), process additional below
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat((bsz.view(1), seq_len.view(1), torch.LongTensor([-1])))
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings

        # Return the weights selected by the positions generated above
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    # Helper function below
    def max_positions(self):
        """
        Maximum number of supported positions.
        """
        return int(1e5)  # an arbitrary large number
