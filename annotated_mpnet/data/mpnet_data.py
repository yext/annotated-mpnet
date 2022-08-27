"""
Module containing the logic for handling the dataset for MPNet including the Dataset class as well
as the data collator
"""

import logging
from typing import Dict, Sized, Iterator
from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)

import os

import numpy as np
import torch
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizer

from annotated_mpnet.utils import utils
from annotated_mpnet.utils.perm_utils_fast import make_span_perm


class MPNetDataset(torch.utils.data.Dataset):
    """
    Class handling the collection of samples for MPNet pretraining
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
    ) -> None:
        """
        Init the dataset

        Args:
            tokenizer: the tokenizer for the model
            file_path: the file path containing the data in text lines to be tokenized
            block_size: the maximum amount of tokens in the block
        """
        super().__init__()

        self.tokenizer = tokenizer

        # Check if the file path exists
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")

        LOGGER.info(f"Creating features from dataset file at {file_path}")

        # We open the file and gather line by line (obviously this means the dataset must fit in
        # memory, just something to keep in mind)
        with open(file_path, encoding="utf-8") as f:
            lines = [
                line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())
            ]

        # Now we process batch encoding using the tokenizer passed in
        # Options are:
        # * Add special tokens in so we can have <s> and </s>
        # * Truncation is true so we can cut any examples to the block size
        # * Truncate at block size by setting max length to the block size
        batch_encoding = tokenizer(
            lines, add_special_tokens=True, truncation=True, max_length=block_size
        )

        # Extract the input IDs and store them in the "examples" dict. We do not need to save the
        # attention mask because it will be created for us in the two-stream self-attention module
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, torch.tensor]:
        return self.examples[i]


class DataCollatorForMaskedPermutedLanguageModeling:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        pred_prob: float = 0.15,
        keep_prob: float = 0.10,
        rand_prob: float = 0.10,
        whole_word_mask: bool = True,
        use_fast: bool = True,
        random_seed=None,
    ) -> None:
        """
        Init the dataset

        Args:
            tokenizer: the tokenizer for the model
            pred_prob: the probability that a token will be in the prediction section
            keep_prob: the probability that a token in the pred will be kept as is
            rand_prob: the probability that a token in the pred will be randomly corrupted
            whole_word_mask: boolean dictating whether or not we should be doing whole word masking
                when generating permutations
            use_fast: boolean dictation whether to use the Cython implementation of a costly
                function to speed up training
            random_seed: used for generating reproducible permutations during testing
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.pred_prob = pred_prob
        self.keep_prob = keep_prob
        self.rand_prob = rand_prob

        self.use_fast = use_fast

        self.random_seed = random_seed

        # Let's also create a byte tensor that maps words that begin with ##
        # We'll use this later on to do whole word masking
        if whole_word_mask:
            self.whole_word_mask_map = torch.ByteTensor(
                [
                    not token.startswith("##")
                    for token, _ in sorted(tokenizer.vocab.items(), key=lambda x: x[1])
                ]
            )
        else:
            self.whole_word_mask_map = None

        # Finally, let's create a weight tensor that will make sure no special tokens are selected
        # when we are corrupting values later on
        weights = np.ones(len(tokenizer.vocab))

        for idx in tokenizer.all_special_ids:
            weights[idx] = 0

        self.weights = weights / weights.sum()

    def __call__(self, examples):
        return self.collate_fn(examples)

    def collate_fn(self, examples):
        """
        The core collating function for MPNet lives here
        """

        # Start by creating a batch
        batch = self.tokenizer.pad(examples, return_tensors="pt")

        # Let's get the input IDs for the batch
        src_tokens = batch["input_ids"]

        # Get inline tokens so that we can keep track of the total non-padding tokens in the batch
        inline_tokens = src_tokens.view(-1)

        # Use these inline tokens with the padding_idx ignored to get the total number of tokens in
        # the batch
        ntokens = inline_tokens[inline_tokens != 1].numel()

        # Let's get the batch dimension
        sz = src_tokens.size()

        # Calculate the pred_size for this batch
        pred_size = round(sz[1] * self.pred_prob)

        # If the sequence is too short to have a masked token, we will mask the whole thing
        if pred_size == 0:
            pred_size = sz[1]

        # If we DO want to mask whole words, we use the span perm function which will permute whole
        # spans / words. Otherwise, we simply do a random permutation
        if self.whole_word_mask_map is not None:
            positions = torch.stack(
                [self.span_perm(src_tokens[i], pred_size) for i in range(sz[0])]
            )
        else:
            positions = torch.stack([torch.randperm(sz[1]) for i in range(sz[0])])

        # Now we actually do the permutation of the inputs based on the position outputs from above
        src_tokens, targets = self.permute_inputs(src_tokens, positions), None

        # Get the range of indices where mask tokens exist
        mask_range = range(sz[1] - pred_size, sz[1])

        # Extract targets, i.e. masked tokens, using the mask range
        targets = src_tokens[:, mask_range].contiguous()

        # Now mask the tokens using the mask_perm function. This will mask tokens and corrupt them
        # at a lower probability
        masked_tokens = self.mask_perm(targets.clone(), self.tokenizer.mask_token_id)

        # Now construct the postions and input IDs using the mask portion
        src_tokens = torch.cat((src_tokens, masked_tokens, masked_tokens), dim=1)
        positions = torch.cat(
            (positions, positions[:, mask_range], positions[:, mask_range]), dim=1
        )

        # Now load these up into collated form
        batch["targets"] = targets
        batch["input_ids"] = src_tokens
        batch["positions"] = positions
        batch["pred_size"] = targets.size(1)
        batch["ntokens"] = ntokens

        return batch

    # Let's define helper functions here
    def span_perm(self, x: torch.Tensor, pred_size=None) -> torch.Tensor:
        """
        This clever function is able to permute the input sequence while ALSO keeping words intact

        Args:
            x: input IDs
            pred_size: the total amount of tokens that will be predicted

        Returns:
            A tensor containing the permuted positions for that input ID sequence while also keeping
            words intact
        """

        # Get a "mask" of which input IDs are the STARTS of words by using the map we created
        # earlier. This essentially uses the input IDs as indices to extract a 1 or 0 from the map
        word_begins_mask = self.whole_word_mask_map.gather(0, x)

        # Now let's get the positional indices of each word beginning. This will help us properly
        # create permutation "spans" later on. These are sorted from lowest to highest, so it might
        # look like:
        # [0, 1, 3, 4, 7, 10]
        # where each index represents the start of a word
        word_begins_idx = word_begins_mask.nonzero().view(-1).tolist()

        # Get the size of the positional sequence
        sz = len(word_begins_idx)

        # If a random seed is passed, we set the random seed before doing the perm below
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Create a permutation based on this size. In the example above, we would permute length
        # 6, thus would have something like:
        # np.array([1 5 3 2 4 0])
        # This will serve as the permuted indices of the positions stored in `word_begins_idx`
        perm = np.random.permutation(sz)

        # We also need to append the total length of the input sequence `x` for a reason that will
        # become clear below. Essentially, you may think about each of the values in
        # word_begins_mask as the start of the "span" while the next index is the end of the "span".
        # In the case of the last span (i.e. 10 in our example above), there would not be an end to
        # the span without appending the length of the input sequence
        word_begins_idx.append(x.size(0))

        # The best practice here will be to use the Cython implementation of the function, but we
        # leave the Python implementation as a branch here so that the reader can fully understand
        # how the permutations are generated without having to read into less friendly Cython
        if self.use_fast:
            # Pass the necessary components into the Cython function and get the ndarray back
            spans = make_span_perm(perm, word_begins_idx, x.size(0))

        else:
            # Finally, we use everything we've just created to actually create the spans below

            # Begin by defining a numpy array of the appropriate length (i.e. the size of the
            # source)
            spans = np.zeros(x.size(0), dtype=np.int64)

            # We create an index tracker called `g` that will only update when we are adding spans
            # to `spans`
            g = 0

            # Now iterate through each of the indices of `word_begins_idx`. Use those indices to
            # extract the permuted indices and use that to build the start and end span.
            #
            # Using the above example for just the first iteration:
            # i = 0
            # perm[i] = 1
            # perm[i] + 1 = 2
            # start = word_begins_idx[1] = 1
            # end = word_begins_idx[2] = 3
            #
            # Now that we have our start and end indices, we create our "span" by iterating over
            # those indices and using our `g` counter to add the span intos the final `spans` holder
            for i in range(len(word_begins_idx) - 1):
                start = word_begins_idx[perm[i]]
                end = word_begins_idx[perm[i] + 1]

                for j in range(start, end):
                    spans[g] = j
                    g += 1

        # Now we do one last shuffle of the masked indices to make sure they are also permuted
        if pred_size is not None:
            # Set the random seed if it's been provided
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            np.random.shuffle(spans[-pred_size:])

        return torch.from_numpy(spans)

    def mask_perm(self, tokens: torch.Tensor, mask_idx: int = None) -> torch.Tensor:
        """
        Masks and corrupts tokens within the predicted portion of the input tokens (i.e. the last n
        tokens, where n = pred_size)

        Args:
            tokens: the tokens in `mask_range` which are the last n tokens where n = pred_size
            mask_idx: the index of the mask token, extracted from the tokenizer

        Returns:
            Returns the tokens with either masked or corrupted indices (or unchanged ones)
        """

        # Extract the probability that a token will be masked. This is what's left over after
        # subtracting the keep_prob and the rand_prob from 1. At default this means mask_prob is
        # 0.80
        mask_prob = 1.0 - self.rand_prob - self.keep_prob

        # Get the corruption probability, which is a function of how rand_prob and keep_prob play
        # with each other
        # You may think of this as the ratio of rand_prob to keep_prob
        # This is the probability that will be used to determine whether or not a token should be
        # randomly corrupted
        corrupt_prob = self.rand_prob / (
            1.0 - mask_prob
        )  # i.e. rand_prob / (rand_prob + keep_prob)

        # Set the torch random seed if one has been provided
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        # Now use torch's builtin bernoulli function to choose tokens (from a bernoulli dist.) that
        # will be masked and save their indices
        # More specifically, a tensor the size of `tokens` is created where each value is mask_prob
        # This is then passed to the bernoulli distribution, which either generates a 0 or 1 based
        # on that mask_prob
        # Finally we convert it to a boolean tensor for reasons that will be clear in the next step
        mask_indices = torch.bernoulli(torch.full(tokens.shape, mask_prob)).bool()

        # Set the torch random seed if one has been provided
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        # Now we get the indices where we want to corrupt the tokens using a similar approach to the
        # above logic. Using corrupt_prob, we use the bernoulli distribution to get a list of
        # indices that we will want to corrupt. We also test against the boolean tensor we made
        # above to make sure that we aren't corrupting any tokens that were already slated for
        # maskin
        corrupt_indices = (
            torch.bernoulli(torch.full(tokens.shape, corrupt_prob)).bool() & ~mask_indices
        )

        # Now we mask and corrupt the tokens dictated by the indices above
        # We use the generate_random_tensor helper function to select random indices from the vocab
        tokens[mask_indices] = mask_idx
        tokens[corrupt_indices] = self.generate_random_tensor(corrupt_indices.sum().tolist()).to(
            tokens.device
        )

        return tokens

    def permute_inputs(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        This function uses the positions we permuted earlier to permute the actual input_ids

        Args:
            inputs: the input_ids
            positions: the permuted positions
        """

        # Get the shape of the inputs, i.e., (batch_size, seq_len)
        sz = inputs.size()

        # Next we get an offset measurement. This is to calculate position permutations if there
        # are multiple source tokens in the batch. The arange function accepts a starting value, in
        # this case 0, and ending value, in this case sz[0] * sz[1] (the total number of values in
        # the batch), and a step size, in this case sz[1], which is the size of each input ID
        #
        # This means that each subsequent set of src_tokens in the batch will be offset by the
        # sequence length. This ensures we can do the permutation calculation properly below
        offset = torch.arange(0, sz[0] * sz[1], sz[1])

        # Now use this offset to amend the position values
        index = positions + offset.unsqueeze_(1)

        return inputs.reshape(-1)[index]

    def generate_random_tensor(self, sz: int) -> torch.Tensor:
        """
        Helper function that will randomly select IDs from the tokenizer vocab to corrupt tokens
        that aren't masked

        Args:
            sz: the number of random tokens to extract
        """
        # Set the numpy random seed if it's been provided
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        return torch.from_numpy(np.random.choice(len(self.tokenizer.vocab), sz, p=self.weights))


class RandomSamplerWithSeed(Sampler[int]):
    """
    Random sampler based on the base Sampler class that allows for seeded random sampling such that
    epochs are reproducible. If a seed isn't provided, training will be truly random.
    """

    def __init__(self, data_source: Sized, epoch: int, random_seed=None) -> None:
        self.data_source = data_source
        self.epoch = epoch

        if random_seed is None:
            self.random_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            self.random_seed = random_seed

    def __iter__(self) -> Iterator[int]:
        with utils.numpy_seed(self.epoch + self.random_seed):
            shuffle = np.random.permutation(len(self.data_source))

        return iter(shuffle)

    def __len__(self) -> int:
        return len(self.data_source)
