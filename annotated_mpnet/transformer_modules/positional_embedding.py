"""
Wrapping function for positional embeddings which allows users to selected learned or sinusoidal
embeddings
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

from annotated_mpnet.transformer_modules import LearnedPositionalEmbedding
from annotated_mpnet.transformer_modules import SinusoidalPositionalEmbedding


def PositionalEmbedding(
    num_embeddings: int, embedding_dim: int, padding_idx: int, learned: bool = False
):
    """
    Wrapping function that will select the appropriate positional embedding type specified by the
    learned parameter
    """

    # If we specified "learned" to be True, we want to create a learned positional embedding module
    if learned:
        # If we specify a padding index, we need to update the total number of embeddings
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1

        # Instantiate the learned positional embeddings
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)

        # Make sure the weights are properly initialized here
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)

        # If we specified a padding index, we need to make sure this weight is zeroed out
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    # Branch to create sinusoidal embeddings if "learned" is False
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim, padding_idx, init_size=num_embeddings + padding_idx + 1
        )

    return m
