import torch
from datasets import load_from_disk
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import BertTokenizerFast

from src.utils.data_utils import (
    PAD,
    START_TAG,
    STOP_TAG,
    TransformerCollatorHF,
    UniformSampler,
    tokenize_dataset,
)


class ActiveLearningDatasetPool:
    """Main class managing the pool for active learning pipeline."""

    def __init__(self, config=None):
        """Initialize the pool for AL pipeline.

        :param config : hydra config
        """
        self.config = config
        self.pool_size = config.activelearning.pool_size

        # ______ data tokenization
        self.dataset = load_from_disk(config.datamodule.dataset_path)
        self.dataset["pool"] = self.dataset["pool"].select(range(self.pool_size))

        print("[WARNING] WE ARE NOT TRUNCATING THE INPUT ")
        self.tokenizer = BertTokenizerFast.from_pretrained(
            config.model.config.embedder_type, add_prefix_space=True
        )

        # ________ Tag Mapping
        tags = self.dataset["train"].features["ner_tags"].feature.names
        tags.insert(0, PAD)
        tags.append(START_TAG)
        tags.append(STOP_TAG)
        self.label2idx = {v: k for k, v in enumerate(tags)}
        self.idx2labels = tags
        try:
            extra_cols = ["tokens", "pos_tags", "chunk_tags", "ner_tags"]
            col_to_del = [col for col in extra_cols if col in self.dataset["train"].column_names]
            self.tokenized_datasets = self.dataset.map(
                tokenize_dataset,
                batched=True,
                fn_kwargs={"tokenizer": self.tokenizer},
                remove_columns=col_to_del,
            )
        except ValueError as e:
            print("Value Error : ", e)
        # ________ Splits
        self.pool_dataset = self.tokenized_datasets["pool"].with_format(type="torch")
        self.validation_dataset = self.tokenized_datasets["validation"].with_format(type="torch")
        self.test_dataset = self.tokenized_datasets["test"].with_format(type="torch")
        self.train_dataset = self.tokenized_datasets["train"].with_format(type="torch")

        # ________ Indices
        self.pooled_indices = self.pool_dataset["id"]
        self.labeled_idxs = torch.zeros(self.pooled_indices.shape, dtype=bool)
        self._init_pool_indices = self.pooled_indices.clone()

        # _______ Collator
        self.collator = TransformerCollatorHF(self.tokenizer)

    def initial_pool_indices(self):
        """Getter for initial pool indices."""
        return self._init_pool_indices

    def get_dataloader(self, split: str, batch_size):
        """
        :param split [str]: ["train","test","validation"]
        :param batch_size int: [batch size in use for the dataloader]
        """
        split_to_ds = {
            "train": self.train_dataset,
            "test": self.test_dataset,
            "validation": self.validation_dataset,
        }
        assert split in split_to_ds
        dataset = split_to_ds[split]
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collator,
            pin_memory=True,
            num_workers=self.config.datamodule.num_workers,
        )
        return dl

    def initialize_labels(self, num):
        """
        :param num [TODO:type]: [TODO:description]
        """
        tmp_idx = torch.randperm(len(self.pooled_indices))
        self.labeled_idxs[tmp_idx[:num]] = True
        return self.pooled_indices[self.labeled_idxs]

    def get_labeled_indices(self):
        """[TODO:description]"""
        labeled_indices = self.pooled_indices[self.labeled_idxs.nonzero().flatten()]
        return labeled_indices

    def get_unlabeled_indices(self):
        unlabeled_indices = self.pooled_indices[(self.labeled_idxs == 0).nonzero().flatten()]
        return unlabeled_indices

    def update_idx(self, sample_id: Tensor):
        """
        sample_id : List[int], with values from "id" column of the dataset
        """
        mask = torch.isin(self._init_pool_indices, sample_id)
        self.labeled_idxs[mask] = True

    def get_labeled_data(self):
        labeled_idx = torch.nonzero(self.labeled_idxs == 1).flatten().tolist()
        return DataLoader(
            self.pool_dataset,
            batch_size=self.config.datamodule.batch_size,
            sampler=SubsetRandomSampler(labeled_idx),
            collate_fn=self.collator,
            pin_memory=True,
            num_workers=self.config.datamodule.num_workers,
        )

    def get_unlabeled_data(self, sequential_sample: bool, batch_size: int):
        unlabeled_indices = (torch.nonzero(self.labeled_idxs == 0).flatten()).tolist()
        if batch_size is None:
            raise ValueError("batch size cannot be None")
        if sequential_sample:
            subset_ds = Subset(self.pool_dataset, unlabeled_indices)
            return DataLoader(
                subset_ds,
                shuffle=False,
                batch_size=batch_size,
                collate_fn=self.collator,
                pin_memory=True,
                num_workers=self.config.datamodule.num_workers,
            )
        else:
            uniform_sampler = UniformSampler(self.pool_dataset, batch_size, unlabeled_indices)
            return DataLoader(
                self.pool_dataset,
                batch_sampler=uniform_sampler,
                collate_fn=self.collator,
                pin_memory=True,
                num_workers=self.config.datamodule.num_workers,
            )
