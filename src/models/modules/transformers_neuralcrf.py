#
# @author: Allan
#
from typing import Tuple

import torch
import torch.nn as nn

from src.models.modules.bilstm_encoder import BiLSTMEncoder
from src.models.modules.fast_linear_crf_inferencer import FastLinearCRF
from src.models.modules.linear_encoder import LinearEncoder
from src.models.modules.transformers_embedder import TransformersEmbedder


class dotdict(dict):
    """dot.notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TransformersCRF(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = dotdict(config)
        self.embedder = TransformersEmbedder(
            transformer_model_name=config.embedder_type,
            parallel_embedder=config.parallel_embedder,
        )
        self.encoder = LinearEncoder(
            label_size=config.label_size,
            input_dim=self.embedder.get_output_dim(),
        )
        self.inferencer = FastLinearCRF(
            label_size=config.label_size,
            label2idx=config.label2idx,
            add_iobes_constraint=config.add_iobes_constraint,
            idx2labels=config.idx2labels,
        )

        # self.inferencer = CRF(config.label_size)

        # self.pad_idx = config.label2idx[PAD]

    # @overrides
    def forward(
        self,
        subword_input_ids: torch.Tensor,
        word_seq_lens: torch.Tensor,
        orig_to_tok_index: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the negative loglikelihood.

        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        :param context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param labels: (batch_size x max_seq_len)
        :return: the total negative log-likelihood loss
        """

        word_rep, _ = self.embedder(subword_input_ids, orig_to_tok_index, attention_mask)
        lstm_scores = self.encoder(word_rep, word_seq_lens)

        batch_size = word_rep.size(0)
        sent_len = word_rep.size(1)
        dev_num = word_rep.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        maskTemp = (
            torch.arange(1, sent_len + 1, dtype=torch.long, device=curr_dev)
            .view(1, sent_len)
            .expand(batch_size, sent_len)
        )
        mask = torch.le(
            maskTemp,
            word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len),
        )
        unlabed_score, labeled_score = self.inferencer(lstm_scores, word_seq_lens, labels, mask)
        return torch.sum(unlabed_score) - labeled_score

    def compute_gradient(
        self,
        subword_input_ids: torch.Tensor,
        word_seq_lens: torch.Tensor,
        orig_to_tok_index: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the negative loglikelihood.

        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        :param context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param labels: (batch_size x max_seq_len)
        :return: the total negative log-likelihood loss
        """
        word_rep, _ = self.embedder(subword_input_ids, orig_to_tok_index, attention_mask)
        lstm_scores = self.encoder(word_rep, word_seq_lens)

        batch_size = word_rep.size(0)
        sent_len = word_rep.size(1)
        dev_num = word_rep.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        maskTemp = (
            torch.arange(1, sent_len + 1, dtype=torch.long, device=curr_dev)
            .view(1, sent_len)
            .expand(batch_size, sent_len)
        )
        mask = torch.le(
            maskTemp,
            word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len),
        )
        unlabed_score, labeled_score = self.inferencer(lstm_scores, word_seq_lens, labels, mask)

        nll = torch.sum(unlabed_score) - labeled_score
        word_rep.retain_grad()
        nll.backward(retain_graph=True)
        # TODO how do you know this implementation is correct  do you know that imp
        return nll, word_rep.grad

    def compute_Z(
        self,
        subword_input_ids: torch.Tensor,
        word_seq_lens: torch.Tensor,
        orig_to_tok_index: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the negative loglikelihood.

        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        :param context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param labels: (batch_size x max_seq_len)
        :return: the total negative log-likelihood loss
        """

        word_rep, _ = self.embedder(subword_input_ids, orig_to_tok_index, attention_mask)
        lstm_scores = self.encoder(word_rep, word_seq_lens)
        batch_size = word_rep.size(0)
        sent_len = word_rep.size(1)
        dev_num = word_rep.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        maskTemp = (
            torch.arange(1, sent_len + 1, dtype=torch.long, device=curr_dev)
            .view(1, sent_len)
            .expand(batch_size, sent_len)
        )
        mask = torch.le(
            maskTemp,
            word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len),
        )
        unlabed_score, _ = self.inferencer(lstm_scores, word_seq_lens, labels, mask)
        return unlabed_score

    def decode(
        self,
        subword_input_ids: torch.Tensor,
        word_seq_lens: torch.Tensor,
        orig_to_tok_index: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode the batch input.

        :param batchInput:
        :return:
        """
        word_rep, last_hidden_state = self.embedder(
            subword_input_ids, orig_to_tok_index, attention_mask
        )
        features = self.encoder(word_rep, word_seq_lens)
        bestScores, decodeIdx = self.inferencer.decode(features, word_seq_lens)
        return (
            bestScores,
            decodeIdx,
            features,
            last_hidden_state,
        )
