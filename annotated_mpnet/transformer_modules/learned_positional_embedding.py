"""
Module containing the LearnedPositionalEmbedding option, which learns position values instead of
something like a sinusoidal distribution
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


class LearnedPositionalEmbedding(nn.Embedding):
    """
    A subclass of the Embedding module that will operate as a layer for learning positional embeds
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int) -> None:
        # Initialize the superclass embedding layer
        super().__init__(num_embeddings, embedding_dim, padding_idx)

        # We set this ONNX variable just in case it breaks something down the line, but I think it's
        # useless for us
        self.onnx_trace = False

    def forward(self, input, incremental_state=None, positions=None):
        """
        Forward pass for the learned embeddings. We pass in a batch of examples in "input" with the
        shape (bsz x seq_len) as expected. One note is that if positions are precomputed (which will
        most likely be the case for MPNet), we should not pass in a padding_idx when instantiating
        this class.
        """

        # Assert that only one of `positions` or `padding_idx` is set
        assert (positions is None) or (
            self.padding_idx is None
        ), "If `positions` is precomputed, do NOT pass in a padding_idx"

        # Let's create the positions if they are not precomputed
        if positions is None:
            # We branch to this "incremental_state" logic only if we're doing ONNX exporting
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = input.data.new(1, 1).fill_(int(self.padding_idx + input.size(1)))
            else:
                # Create positions using the `make_positions` function. This basically just creates
                # incremental positions starting at padding_idx+1. Very simple function that you
                # can check in the utils package
                positions = utils.make_positions(
                    input.data,
                    self.padding_idx,
                    onnx_trace=self.onnx_trace,
                )

        # Do the actual embedding pass here now
        return super().forward(positions)

    # Below are some convenience functions and aliases for this class. Should not be of too much
    # importance for our usage
    def max_positions(self):
        """
        Returns the max number of positional embeddings
        """
        if self.padding_idx is not None:
            return self.num_embeddings - self.padding_idx - 1
        else:
            return self.num_embeddings

    def _forward(self, positions):
        """
        Alias for calling the embedding layer if positions are precomputed
        """
        return super().forward(positions)
