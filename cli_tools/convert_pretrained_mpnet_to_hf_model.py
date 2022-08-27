"""
Script for converting our pretrained models to the main MPNet class in Huggingface (which is 
MPNetForMaskedLM). This is ported over from Kaitao Song's Github, who was the lead author for the 
MPNet paper. We have amended a good amount of the notation to fit our construction of 
MPNetForPretraining, as well as commenting each step of the logic so that posterity knows what we're
doing here.
"""

import logging
from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)

import argparse
import pathlib

import torch

from transformers import MPNetConfig, MPNetForMaskedLM
from transformers.utils import logging as hf_logging
from transformers.models.mpnet import MPNetLayer

# Set up the transformers logger. This will help us track down any errors while loading the weights
HF_LOGGER = hf_logging.get_logger(__name__)
hf_logging.set_verbosity_info()


def convert_mpnet_checkpoint_to_pytorch(
    mpnet_checkpoint_path: str, pytorch_dump_folder_path: str
) -> None:
    """
    This is the main function of the script. It takes in a checkpoint path pointing to a specific
    .pt serialization of the pretrained MPNet model and dumps the necessary files for the model to
    be in accordance with Huggingface specifications.

    Args:
        mpnet_checkpoint_path: the path to the .pt model containing the model weights and args
        pytorch_dump_folder_path: the path to the directory that will contain the necessary
            components for the HF model
    """

    # Load up the state dicts (one for the weights and one for the args) from the provided
    # serialization path
    state_dicts = torch.load(mpnet_checkpoint_path)

    # Extract the model args so that we can properly set the config later on
    # Extract the weights so we can set them within the constructs of the model
    mpnet_args = state_dicts["args"]
    mpnet_weight = state_dicts["model_states"]

    # Now we use the args (and one componennt of the weight to get the vocab size) to set the
    # MPNetConfig object, which will properly instantiate the MPNetForMaskedLM model to the specs
    # we set when we pretrained the model
    config = MPNetConfig(
        vocab_size=mpnet_weight["sentence_encoder.embed_tokens.weight"].size(0),
        hidden_size=mpnet_args.encoder_embed_dim,
        num_hidden_layers=mpnet_args.encoder_layers,
        num_attention_heads=mpnet_args.encoder_attention_heads,
        intermediate_size=mpnet_args.encoder_ffn_dim,
        # One note here is that, although we specify max_positions as some int that is usually the
        # same size as max_tokens, since position embeddings start at padding_idx + 1, we need to
        # add 2 to the final count, since the lowest possible position is 2
        max_position_embeddings=mpnet_args.max_positions + 2,
        hidden_act=mpnet_args.activation_fn,
        layer_norm_eps=1e-5,
    )

    # Now load the model with randomized weights
    model = MPNetForMaskedLM(config)

    # Extract the tensor representing the the word embeddings from the HF model. This will help us
    # appropriately type our tensors that are being extracted from the state_dict. A type mismatch
    # will lead to issues when loading up the HF model
    tensor = model.mpnet.embeddings.word_embeddings.weight

    # Extract the hidden size, this will help us set the appropriate weights for the self-attention
    # mechanism later on. When we pretrained it, we stored the QKV weights as one single layer, so
    # we need to break that up into chunks of even size (config.hidden_size)
    dim = config.hidden_size

    # Set the embeddings portion of the HF model. This should be pretty straightforward, we're
    # simply matching up the HF names with our names. Of course, as pointed out above, we are also
    # explicitly setting the type of our weights to the type of the MPNet model so that we don't
    # have any type mismatches
    model.mpnet.embeddings.word_embeddings.weight.data = mpnet_weight[
        "sentence_encoder.embed_tokens.weight"
    ].type_as(tensor)
    model.mpnet.embeddings.position_embeddings.weight.data = mpnet_weight[
        "sentence_encoder.embed_positions.weight"
    ].type_as(tensor)
    model.mpnet.embeddings.LayerNorm.weight.data = mpnet_weight[
        "sentence_encoder.emb_layer_norm.weight"
    ].type_as(tensor)
    model.mpnet.embeddings.LayerNorm.bias.data = mpnet_weight[
        "sentence_encoder.emb_layer_norm.bias"
    ].type_as(tensor)

    # Here, we're setting the weights and biases for the LM head. This is important for loading into
    # the base HF model type (MPNetForMaskedLM), but this will usually be discarded in any sort of
    # downstream task in favor of the appropriate fine-tuning head
    model.lm_head.dense.weight.data = mpnet_weight["lm_head.dense.weight"].type_as(tensor)
    model.lm_head.dense.bias.data = mpnet_weight["lm_head.dense.bias"].type_as(tensor)
    model.lm_head.layer_norm.weight.data = mpnet_weight["lm_head.layer_norm.weight"].type_as(tensor)
    model.lm_head.layer_norm.bias.data = mpnet_weight["lm_head.layer_norm.bias"].type_as(tensor)
    model.lm_head.decoder.weight.data = mpnet_weight["lm_head.weight"].type_as(tensor)
    model.lm_head.decoder.bias.data = mpnet_weight["lm_head.bias"].type_as(tensor)

    # Match up the relative attention bias weights with each other
    model.mpnet.encoder.relative_attention_bias.data = mpnet_weight[
        "sentence_encoder.relative_attention_bias.weight"
    ].type_as(tensor)

    # Now for each SentenceEncoderLayer we have, we need to load the weights appropriately
    # Each iteration is simply matching up the weights and biases for each internal component of the
    # current layer
    for i in range(config.num_hidden_layers):
        # Get the layer of the HF model
        layer: MPNetLayer = model.mpnet.encoder.layer[i]

        # Get the prefix for each of our layers. This will keep us from having to rewrite it every
        # time below
        prefix = f"sentence_encoder.layers.{i}."

        # Match up the weight and bias for the initial projection layer of the self-attention
        # mechanism (i.e. stacked QKV)
        in_proj_weight = mpnet_weight[prefix + "self_attn.in_proj_weight"].type_as(tensor)
        in_proj_bias = mpnet_weight[prefix + "self_attn.in_proj_bias"].type_as(tensor)

        # Now, as described above when we stored the hidden size in `dim`, we need to chunk up the
        # weights so that we can get the separate Q, K, and V weights
        #
        # As an example, if we had a hidden size of 768 (which is by far the most common), we would
        # find that:
        # Q -> corresponds to the weights/biases from 0 - 768 (not inclusive)
        # K -> corresponds to the weights/biases from 768 - 1536 (not inclusive)
        # V -> corresponds to the weights/biases from 1536 - 2304 (not inclusive)
        # So now the weights are chunked out
        layer.attention.attn.q.weight.data = in_proj_weight[dim * 0 : dim * 1]
        layer.attention.attn.k.weight.data = in_proj_weight[dim * 1 : dim * 2]
        layer.attention.attn.v.weight.data = in_proj_weight[dim * 2 : dim * 3]
        layer.attention.attn.q.bias.data = in_proj_bias[dim * 0 : dim * 1]
        layer.attention.attn.k.bias.data = in_proj_bias[dim * 1 : dim * 2]
        layer.attention.attn.v.bias.data = in_proj_bias[dim * 2 : dim * 3]

        # Extract and match up the out projection layer as well as the LayerNorm right after
        # If normalize_before were set to True in the original pretraining script, it probably
        # won't have any effect since the MPNetForMaskedLM class will always load the LayerNorm
        # weights AFTER the self-attention component
        layer.attention.attn.o.weight.data = mpnet_weight[
            prefix + "self_attn.out_proj.weight"
        ].type_as(tensor)
        layer.attention.attn.o.bias.data = mpnet_weight[prefix + "self_attn.out_proj.bias"].type_as(
            tensor
        )
        layer.attention.LayerNorm.weight.data = mpnet_weight[
            prefix + "self_attn_layer_norm.weight"
        ].type_as(tensor)
        layer.attention.LayerNorm.bias.data = mpnet_weight[
            prefix + "self_attn_layer_norm.bias"
        ].type_as(tensor)

        # Extract the weights and biases for the feed-forward net after the self-attention
        # calculation
        layer.intermediate.dense.weight.data = mpnet_weight[prefix + "fc1.weight"].type_as(tensor)
        layer.intermediate.dense.bias.data = mpnet_weight[prefix + "fc1.bias"].type_as(tensor)
        layer.output.dense.weight.data = mpnet_weight[prefix + "fc2.weight"].type_as(tensor)
        layer.output.dense.bias.data = mpnet_weight[prefix + "fc2.bias"].type_as(tensor)

        # Extract the final LayerNorm and set it here
        layer.output.LayerNorm.weight.data = mpnet_weight[
            prefix + "final_layer_norm.weight"
        ].type_as(tensor)
        layer.output.LayerNorm.bias.data = mpnet_weight[prefix + "final_layer_norm.bias"].type_as(
            tensor
        )

    # Create the dump directory if it doesn't exist
    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Saving model to {pytorch_dump_folder_path}")

    # Now that the config and weights are loaded into the model class, we can use HF's builtin
    # save_pretrained function to dump the appropriate contents to the provided dir path
    model.save_pretrained(pytorch_dump_folder_path)


def cli_main():
    """
    Wrapper function so we can define a CLI entrypoint when setting up this package
    """
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--mpnet-checkpoint-path",
        default=None,
        type=str,
        required=True,
        help="Path to our MPNet checkpoint",
    )
    parser.add_argument(
        "--hf-model-folder-path",
        default=None,
        type=str,
        required=True,
        help="Path to dump the newly built Huggingface model",
    )
    args = parser.parse_args()
    convert_mpnet_checkpoint_to_pytorch(
        args.mpnet_checkpoint_path,
        args.hf_model_folder_path,
    )


if __name__ == "__main__":
    cli_main()
