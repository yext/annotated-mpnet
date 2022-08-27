"""
Fairseq extension of LayerNorm which trys to use FusedLayerNorm if available
"""

import torch


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    """
    Wrapper function for the torch LayerNorm that tries to use FusedLayerNorm if available
    """

    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass

    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
