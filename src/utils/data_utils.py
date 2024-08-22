import os
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import colored
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Sampler, Subset
from tqdm import tqdm

from src.utils.instance import Instance

B_PREF = "B-"
I_PREF = "I-"
S_PREF = "S-"
E_PREF = "E-"
Other_Tag = "O"

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD = "<PAD>"
UNK = "<UNK>"


import torch


class TransformerCollatorHF:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """pads the input and return a dict with all the tensors."""
        # code from allan j implementation
        word_seq_len = [len(feature["orig_to_tok_index"]) for feature in batch]
        max_seq_len = max(word_seq_len)
        # max length of sentence in the batch
        max_wordpiece_length = max(len(feature["input_ids"]) for feature in batch)
        # max sentence length in  the input ids.
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature["input_ids"])
            # input_ids = feature["input_ids"] + [self.tokenizer.pad_token_id] * padding_length

            input_ids = F.pad(
                feature["input_ids"],
                value=self.tokenizer.pad_token_id,
                pad=(0, padding_length),
            )
            mask = F.pad(feature["attention_mask"], value=0, pad=(0, padding_length))
            type_ids = F.pad(
                feature["token_type_ids"],
                value=self.tokenizer.pad_token_type_id,
                pad=(0, padding_length),
            )
            padding_word_len = max_seq_len - len(feature["orig_to_tok_index"])
            orig_to_tok_index = F.pad(
                feature["orig_to_tok_index"],
                value=0,
                pad=(0, padding_word_len),
            )
            # Comment : transformers by default considers 0 as padding token
            # there in label mapping pad is added at start of tags
            # each label is shiffed by 1
            feature["label_ids"] = feature["label_ids"] + 1
            label_ids = F.pad(feature["label_ids"], value=0, pad=(0, padding_word_len))
            batch[i] = {
                "input_ids": input_ids,
                "attention_mask": mask,
                "token_type_ids": type_ids,
                "orig_to_tok_index": orig_to_tok_index,
                "word_seq_len": feature["word_seq_len"],
                "label_ids": label_ids,
                "id": feature["id"],
                # for testing purpose
                # "test_scores":int(feature["test_scores"]),
            }
        encoded_inputs = {
            key: torch.stack([example[key] for example in batch]) for key in batch[0].keys()
        }
        return encoded_inputs


def tokenize_dataset(examples, tokenizer):
    """"""
    # ___________ validatation __________
    assert tokenizer is not None  # prefer a instance check of the class
    try:
        """tokens.""" in examples
        "ner_tags" in examples
    except KeyError as e:
        print("Dataset must contain 'tokens' and 'ner_tags' columns")
        raise e
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True)

    word_seq = []
    orig_to_tok_list = []
    for idx, _ in enumerate(examples["tokens"]):
        subword_idx2word_idx = tokenized_inputs[idx].word_ids
        orig_to_tok_index = []
        prev_word_idx = -1
        for i, mapped_word_idx in enumerate(subword_idx2word_idx):
            """
            Note: by default, we use the first wordpiece/subword token to represent the word
            If you want to do something else (e.g., use last wordpiece to represent), modify them here.
            """
            if mapped_word_idx is None:  # cls and sep token
                continue
            if mapped_word_idx != prev_word_idx:
                # because we take the first subword to represent the whold word
                orig_to_tok_index.append(i)
                prev_word_idx = mapped_word_idx
        word_seq_len = len(orig_to_tok_index)
        word_seq.append(word_seq_len)
        orig_to_tok_list.append(orig_to_tok_index)
        # print(word_seq_len)
    tokenized_inputs["word_seq_len"] = word_seq
    tokenized_inputs["orig_to_tok_index"] = orig_to_tok_list
    tokenized_inputs["label_ids"] = examples["ner_tags"]
    tokenized_inputs["id"] = [int(x) for x in examples["id"]]
    return tokenized_inputs


def build_emb_table(pretrained_embedding, embedding_dim, word2idx: Dict[str, int]) -> None:
    """
    build the pretrained_embedding table with pretrained word embeddings (if given otherwise, use random embeddings)
    :return:
    """
    print("Building the pretrained_embedding table for vocabulary...")
    scale = np.sqrt(3.0 / embedding_dim)
    if pretrained_embedding is not None:
        print(
            "[Info] Use the pretrained word pretrained_embedding to initialize: %d x %d"
            % (len(word2idx), embedding_dim)
        )
        word_embedding = np.empty([len(word2idx), embedding_dim])
        for word in word2idx:
            if word in pretrained_embedding:
                word_embedding[word2idx[word], :] = pretrained_embedding[word]
            elif word.lower() in pretrained_embedding:
                word_embedding[word2idx[word], :] = pretrained_embedding[word.lower()]
            else:
                # word_embedding[word2idx[word], :] = pretrained_embedding[UNK]
                word_embedding[word2idx[word], :] = np.random.uniform(
                    -scale, scale, [1, embedding_dim]
                )
        pretrained_embedding = None  # remove the pretrained pretrained_embedding to save memory.
    else:
        word_embedding = np.empty([len(word2idx), embedding_dim])
        for word in word2idx:
            word_embedding[word2idx[word], :] = np.random.uniform(
                -scale, scale, [1, embedding_dim]
            )
    return word_embedding


def read_pretrain_embedding(
    embedding_file, embedding_dim
) -> Tuple[Union[Dict[str, np.array], None], int]:
    """
    Read the pretrained word embeddings, return the complete embeddings and the embedding dimension
    :return:
    """
    print("reading the pretraing embedding: %s" % (embedding_file))
    if embedding_file is None:
        print("pretrain embedding in None, using random embedding")
        return None, embedding_dim
    else:
        exists = os.path.isfile(embedding_file)
        if not exists:
            print(
                colored(
                    "[Warning] pretrain embedding file not exists, using random embedding",
                    "red",
                )
            )
            return None, embedding_dim
            # raise FileNotFoundError("The embedding file does not exists")
    embedding_dim = -1
    embedding = dict()
    with open(embedding_file, encoding="utf-8") as file:
        for line in tqdm(file.readlines()):
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedding_dim < 0:
                embedding_dim = len(tokens) - 1
            else:
                # print(tokens)
                # print(embedding_dim)
                assert embedding_dim + 1 == len(tokens)
            embed = np.empty([1, embedding_dim])
            embed[:] = tokens[1:]
            first_col = tokens[0]
            embedding[first_col] = embed
    return embedding, embedding_dim


def convert_iobes(labels: List[str]) -> List[str]:
    """Use IOBES tagging schema to replace the IOB tagging schema in the instance.

    :param insts:
    :return:
    """
    for pos in range(len(labels)):
        curr_entity = labels[pos]
        if pos == len(labels) - 1:
            if curr_entity.startswith(B_PREF):
                labels[pos] = curr_entity.replace(B_PREF, S_PREF)
            elif curr_entity.startswith(I_PREF):
                labels[pos] = curr_entity.replace(I_PREF, E_PREF)
        else:
            next_entity = labels[pos + 1]
            if curr_entity.startswith(B_PREF):
                if next_entity.startswith(Other_Tag) or next_entity.startswith(B_PREF):
                    labels[pos] = curr_entity.replace(B_PREF, S_PREF)
            elif curr_entity.startswith(I_PREF):
                if next_entity.startswith(Other_Tag) or next_entity.startswith(B_PREF):
                    labels[pos] = curr_entity.replace(I_PREF, E_PREF)
    return labels


def build_label_idx(
    insts: List[Instance],
) -> Tuple[List[str], Dict[str, int]]:
    """Build the mapping from label to index and index to labels.

    :param insts: list of instances.
    :return:
    """
    label2idx = {}
    idx2labels = []
    label2idx[PAD] = len(label2idx)
    idx2labels.append(PAD)
    for inst in insts:
        for label in inst.labels:
            if label not in label2idx:
                idx2labels.append(label)
                label2idx[label] = len(label2idx)

    label2idx[START_TAG] = len(label2idx)
    idx2labels.append(START_TAG)
    label2idx[STOP_TAG] = len(label2idx)
    idx2labels.append(STOP_TAG)
    label_size = len(label2idx)
    print(f"#labels: {label_size}")
    print(f"label 2idx: {label2idx}")
    return idx2labels, label2idx


def check_all_labels_in_dict(insts: List[Instance], label2idx: Dict[str, int]):
    for inst in insts:
        for label in inst.labels:
            if label not in label2idx:
                raise ValueError(
                    f"The label {label} does not exist in label2idx dict. The label might not appear in the training set."
                )


def build_word_idx(
    trains: List[Instance], devs: List[Instance], tests: List[Instance]
) -> Tuple[Dict, List, Dict, List]:
    """Build the vocab 2 idx for all instances.

    :param train_insts:
    :param dev_insts:
    :param test_insts:
    :return:
    """
    word2idx = dict()
    idx2word = []
    word2idx[PAD] = 0
    idx2word.append(PAD)
    word2idx[UNK] = 1
    idx2word.append(UNK)

    char2idx = {}
    idx2char = []
    char2idx[PAD] = 0
    idx2char.append(PAD)
    char2idx[UNK] = 1
    idx2char.append(UNK)

    # extract char on train, dev, test
    for inst in trains + devs + tests:
        for word in inst.words:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                idx2word.append(word)
    # extract char only on train (doesn't matter for dev and test)
    for inst in trains:
        for word in inst.words:
            for c in word:
                if c not in char2idx:
                    char2idx[c] = len(idx2char)
                    idx2char.append(c)
    return word2idx, idx2word, char2idx, idx2char


def check_all_obj_is_None(objs):
    for obj in objs:
        if obj is not None:
            return False
    return [None] * len(objs)


class UniformSampler(Sampler):
    def __init__(self, dataset, batch_size, subset_indices=None):
        if subset_indices is None:
            self.sorted_indices = list(np.argsort(dataset["word_seq_len"].tolist()).astype(int))
        else:
            subset_ds = Subset(dataset, subset_indices)
            len_dict = {
                int(ind): len(item["input_ids"]) for ind, item in zip(subset_indices, subset_ds)
            }
            sorted_dcc = {k: v for k, v in sorted(len_dict.items(), key=lambda item: item[1])}
            self.sorted_indices = list(sorted_dcc.keys())
        self.sorted_indices = [int(x) for x in self.sorted_indices]
        self.batch_size = batch_size

    def __iter__(self):
        # idea from https://gist.github.com/pommedeterresautee/1a334b665710bec9bb65965f662c94c8
        indices = deepcopy(self.sorted_indices)
        batch_ordered_sentences = list()
        while len(indices) > 0:
            to_take = min(self.batch_size, len(indices))
            # random select a starting point
            select = np.random.randint(0, (len(indices) - to_take) + 1)
            # select contagious pairs of samples
            batch_ordered_sentences.append(indices[select : select + to_take])
            del indices[select : select + to_take]

        return iter(batch_ordered_sentences)

    def __len__(self):
        return len(self.sorted_indices)
