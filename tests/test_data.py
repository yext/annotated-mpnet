"""
Tests for the collator function in the mpnet_data module
"""

import unittest

import torch
from transformers import AutoTokenizer

from annotated_mpnet.data.mpnet_data import (
    DataCollatorForMaskedPermutedLanguageModeling,
    RandomSamplerWithSeed,
)


class TestData(unittest.TestCase):
    def setUp(self):
        self.examples = [
            {
                "input_ids": torch.tensor(
                    [0, 2027, 2007, 2023, 2746, 2001, 1041, 2311, 3235, 1003, 2],
                ),
            },
            {
                "input_ids": torch.tensor(
                    [0, 2027, 2007, 2182, 3235, 2],
                )
            },
        ]

        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")

        self.collator = DataCollatorForMaskedPermutedLanguageModeling(
            tokenizer=tokenizer, random_seed=12345
        )

    def test_permuted_batch(self):
        correct_return = {
            "input_ids": torch.tensor(
                [
                    [
                        2311,
                        0,
                        2023,
                        1041,
                        2,
                        1003,
                        3235,
                        2746,
                        2027,
                        2007,
                        2001,
                        2007,
                        2001,
                        2007,
                        2001,
                    ],
                    [1, 0, 2182, 1, 1, 1, 1, 3235, 2027, 2007, 2, 2007, 30526, 2007, 30526],
                ]
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]
            ),
            "targets": torch.tensor([[2007, 2001], [2007, 2]]),
            "positions": torch.tensor(
                [
                    [7, 0, 3, 6, 10, 9, 8, 4, 1, 2, 5, 2, 5, 2, 5],
                    [7, 0, 3, 6, 10, 9, 8, 4, 1, 2, 5, 2, 5, 2, 5],
                ]
            ),
            "pred_size": 2,
            "ntokens": 17,
        }

        permuted_examples = self.collator.collate_fn(self.examples)

        self.assertTrue(
            torch.equal(permuted_examples["input_ids"], correct_return["input_ids"]),
            "Input IDs were not permuted correctly",
        )
        self.assertTrue(
            torch.equal(permuted_examples["positions"], correct_return["positions"]),
            "Positions were not permuted correctly",
        )
        self.assertTrue(
            torch.equal(permuted_examples["targets"], correct_return["targets"]),
            "Language modeling targets aren't correct",
        )

    def test_training_seeded_sampling(self):
        sampler_epoch_0 = RandomSamplerWithSeed([self.examples] * 10, epoch=0, random_seed=12345)
        sampler_epoch_1 = RandomSamplerWithSeed([self.examples] * 10, epoch=1, random_seed=12345)

        self.assertEqual(
            list(sampler_epoch_0.__iter__()),
            [0, 7, 3, 9, 6, 4, 1, 8, 5, 2],
            "Sampler not seeding correctly",
        )
        self.assertNotEqual(
            list(sampler_epoch_0.__iter__()),
            list(sampler_epoch_1.__iter__()),
            "Sampler seed may be broken since epoch 0 and epoch 1 are showing the same smpl order",
        )
