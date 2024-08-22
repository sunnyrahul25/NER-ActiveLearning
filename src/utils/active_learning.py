from dataclasses import dataclass
from typing import List, Optional, Type

import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import entropy
from seqeval.metrics import classification_report
from seqeval.metrics.v1 import check_consistent_length
from seqeval.scheme import IOB2, Entities, Token
from tqdm import tqdm

from src.query_strategies.alps import ALPS
from src.query_strategies.badge import BADGE
from src.query_strategies.bald import BALD
from src.query_strategies.contrastive import Contrastive
from src.query_strategies.entropy import Entropy
from src.query_strategies.least_confidence import LeastConfidence
from src.query_strategies.least_token import LeastTokenProbability
from src.query_strategies.long_strategy import LongStrategy
from src.query_strategies.normalized_least_confidence import (
    NormalizedLeastConfidenceStrategy,
)
from src.query_strategies.random_sampling import RandomSampling


def get_strategy(strategy):
    """Factory class for strategy."""
    if strategy == "long":
        return LongStrategy()
    if strategy == "random":
        return RandomSampling()
    if strategy == "lc":
        return LeastConfidence()
    if strategy == "contrastive":
        return Contrastive()
    if strategy == "entropy":
        return Entropy()
    if strategy == "bald":
        return BALD()
    if strategy == "badge":
        return BADGE()
    if strategy == "alps":
        return ALPS()
    if strategy == "nlc":
        return NormalizedLeastConfidenceStrategy()
    if strategy == "ltp":
        return LeastTokenProbability()
    raise NotImplementedError


@dataclass
class TransformerCRFResult:
    scores: torch.Tensor
    embeddings: torch.Tensor
    probs: torch.Tensor
    best_paths: torch.Tensor
    seq_len: List
    batch_indices: torch.IntTensor
    labels: torch.Tensor


def compute_tp_fp_fn(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    scheme: Optional[Type[Token]] = None,
    suffix: bool = False,
):
    """
    from seqeval.scheme import IOB2
    pred = ["B-ORG","O","O"]
    target = ["O","O","O"]
    stats=compute_tp_fp_fn (y_true=target,y_pred=pred,scheme=IOB2)
    for k,v in stats.items():
        print(k,v)
    """
    check_consistent_length(y_true, y_pred)
    entities_true = Entities(y_true, scheme, suffix)
    entities_pred = Entities(y_pred, scheme, suffix)
    target_names = sorted(entities_true.unique_tags | entities_pred.unique_tags)
    stats = {}
    for type_name in target_names:
        entities_true_type = entities_true.filter(type_name)
        entities_pred_type = entities_pred.filter(type_name)
        if type_name not in stats:
            stats[type_name] = {}
            stats[type_name]["TP"] = len(entities_true_type & entities_pred_type)
            stats[type_name]["FP"] = len(entities_pred_type) - stats[type_name]["TP"]
            stats[type_name]["FN"] = len(entities_true_type) - stats[type_name]["TP"]
    return stats


def compute_entropy_vectorized(
    unary_potential, pairwise_potential, word_seq_len, start_tag, end_tag
):
    """Usage :

    from numpy.random import default_rng
    B,N,K = 3,4,5
    upb=default_rng(42).random((B,N,K))
    pp=default_rng(42).random((K,K))
    stag=3
    etag=4
    print(compute_entropy_vectorized(upb,pp,N,stag,etag))
    """
    # TODO is it really necessary to copy maybe get rid of it.

    B, N, K = unary_potential.shape
    H_alphas = np.zeros((B, N, K))
    C = np.zeros((B, N, K))
    unary_potential_vec = np.copy(unary_potential)
    unary_potential_vec[:, 0, :] = unary_potential_vec[:, 0, :] + pairwise_potential[start_tag, :]
    for i, sen_len in enumerate(word_seq_len):
        unary_potential_vec[i, sen_len - 1, :] = (
            unary_potential_vec[i, sen_len - 1, :] + pairwise_potential[:, end_tag]
        )
    lsp = logsumexp(unary_potential_vec[:, 0, :], axis=1)
    C[:, 0, :] = unary_potential_vec[:, 0, :] - np.expand_dims(lsp, axis=1)
    H_alphas = np.zeros((B, N, K))
    for t in range(1, N):
        d = np.zeros((B, K), dtype=np.float64)
        for k in range(0, K):
            aa = (
                C[:, t - 1, :]
                + np.expand_dims(unary_potential_vec[:, t, k], axis=1)
                + pairwise_potential[:, k]
            )
            d[:, k] = np.log(np.sum(np.exp(aa), axis=1))
        d = d - np.expand_dims(logsumexp(d, axis=1), axis=1)
        for j in range(0, K):
            C[:, t, j] = d[:, j]
            pp = pairwise_potential
            other = np.expand_dims(logsumexp(pp[:, j] + C[:, t - 1, :], axis=1), axis=1)
            log_prob = pp[:, j] + C[:, t - 1, :] - other
            exp_log_prob = np.exp(log_prob)
            h_next = (H_alphas[:, t - 1, :] * exp_log_prob).sum(
                axis=1
            )  # may be i can convert later part to log space as well
            H_alphas[:, t, j] = h_next + entropy(exp_log_prob, axis=1)
    entropy_vec = np.zeros(B)
    for i, sen_len in enumerate(word_seq_len):
        entropy_vec[i] = np.sum(
            H_alphas[i, sen_len - 1, :] * np.exp(C[i, sen_len - 1, :])
        ) + entropy(np.exp(C[i, sen_len - 1, :]))
    return entropy_vec


def compute_kl_log_vectorized(
    unary_potential,
    unary_potential_q,
    pairwise_potential,
    sen_len_p,
    sen_len_q,
    start_tag,
    end_tag,
):
    """Usage :

    from numpy.random import default_rng
    B,N,K = 3,4,5
    upb=default_rng(42).random((B,N,K))
    pp=default_rng(42).random((K,K))
    stag=3
    etag=4
    print(compute_entropy_vectorized(upb,pp,N,stag,etag))
    """

    # Variables
    # ==================================
    word_seq_len = np.maximum(sen_len_p, sen_len_q)  # take the minimum of the length of sentence
    B, _, _ = unary_potential.shape
    N = np.max(word_seq_len)
    K = pairwise_potential.shape[0]
    H_p = np.zeros((B, N, K))
    C_p = np.zeros((B, N, K))
    C_q = np.zeros((B, N, K))

    # TODO is it really necessary to copy maybe get rid of it.
    up_p = np.copy(unary_potential)
    up_p[:, 0, :] = up_p[:, 0, :] + pairwise_potential[start_tag, :]

    up_q = np.copy(unary_potential_q)
    up_q[:, 0, :] = up_q[:, 0, :] + pairwise_potential[start_tag, :]

    for i, sen_len in enumerate(word_seq_len):
        up_p[i, sen_len - 1, :] = up_p[i, sen_len - 1, :] + pairwise_potential[:, end_tag]
        up_q[i, sen_len - 1, :] = up_q[i, sen_len - 1, :] + pairwise_potential[:, end_tag]

    lsp_p = logsumexp(up_p[:, 0, :], axis=1)
    lsp_q = logsumexp(up_q[:, 0, :], axis=1)

    C_p[:, 0, :] = up_p[:, 0, :] - np.expand_dims(lsp_p, axis=1)
    C_q[:, 0, :] = up_q[:, 0, :] - np.expand_dims(lsp_q, axis=1)

    for t in range(1, N):
        d_p = np.zeros((B, K), dtype=np.float64)
        d_q = np.zeros((B, K), dtype=np.float64)

        for k in range(0, K):
            temp_p = (
                C_p[:, t - 1, :] + np.expand_dims(up_p[:, t, k], axis=1) + pairwise_potential[:, k]
            )
            temp_q = (
                C_q[:, t - 1, :] + np.expand_dims(up_q[:, t, k], axis=1) + pairwise_potential[:, k]
            )
            d_p[:, k] = np.log(np.sum(np.exp(temp_p), axis=1))
            d_q[:, k] = np.log(np.sum(np.exp(temp_q), axis=1))

        d_p = d_p - np.expand_dims(logsumexp(d_p, axis=1), axis=1)
        d_q = d_q - np.expand_dims(logsumexp(d_q, axis=1), axis=1)

        for j in range(0, K):
            h_next = 0
            C_p[:, t, j] = d_p[:, j]
            C_q[:, t, j] = d_q[:, j]
            pp = pairwise_potential
            other_p = np.expand_dims(logsumexp(pp[:, j] + C_p[:, t - 1, :], axis=1), axis=1)
            other_q = np.expand_dims(logsumexp(pp[:, j] + C_q[:, t - 1, :], axis=1), axis=1)
            log_prob_p = pp[:, j] + C_p[:, t - 1, :] - other_p
            log_prob_q = pp[:, j] + C_q[:, t - 1, :] - other_q
            exp_log_prob_p = np.exp(log_prob_p)
            exp_log_prob_q = np.exp(log_prob_q)
            h_next = (H_p[:, t - 1, :] * exp_log_prob_p).sum(
                axis=1
            )  # may be i can convert later part to log space as well
            H_p[:, t, j] = h_next + entropy(exp_log_prob_p, exp_log_prob_q, axis=1)

    kl_div = np.zeros(B)
    for i, sen_len in enumerate(word_seq_len):
        kl_div[i] = np.sum(H_p[i, sen_len - 1, :] * np.exp(C_p[i, sen_len - 1, :])) + entropy(
            np.exp(C_p[i, sen_len - 1, :]), np.exp(C_q[i, sen_len - 1, :])
        )
    return kl_div


def compute_entropy_log(unary_potential, pairwise_potential, N, start_tag, end_tag):

    # based on  Efficient Computation of the Hidden Markov Model Entropy for a Given Observation Sequence
    # unary , pairwise potential are in log space
    # by default crf maintains potential in logspace
    # N is the length of sentences

    K = pairwise_potential.shape[0]
    H_alphas = np.zeros((N, K))
    C = np.zeros((N, K))
    unary_potential = np.copy(unary_potential)
    unary_potential[0, :] = unary_potential[0, :] + pairwise_potential[start_tag, :]
    unary_potential[N - 1, :] = unary_potential[N - 1, :] + pairwise_potential[:, end_tag]
    C[0, :] = unary_potential[0, :] - logsumexp(unary_potential[0, :])

    for t in range(1, N):
        d = np.zeros(K, dtype=np.float64)
        for k in range(0, K):
            d[k] = np.log(
                np.sum(np.exp(C[t - 1, :] + pairwise_potential[:, k] + unary_potential[t, k]))
            )
        d = d - logsumexp(d)
        for j in range(0, K):
            C[t, j] = d[j]
            h_next = 0
            logprob = (
                pairwise_potential[:, j]
                + C[t - 1, :]
                - logsumexp(pairwise_potential[:, j] + C[t - 1, :])
            )
            h_next = (
                H_alphas[t - 1, :] * np.exp(logprob)
            ).sum()  # may be i can convert later part to log space as well
            H_alphas[t, j] = h_next + entropy(np.exp(logprob))
    return np.sum(H_alphas[N - 1, :] * np.exp(C[N - 1, :])) + entropy(np.exp(C[N - 1, :]))


def compute_kl_log(
    unary_potential,
    unary_potential_q,
    pairwise_potential,
    s,
    t,
    start_tag,
    end_tag,
):
    # based on  Efficient Computation of the Hidden Markov Model Entropy for a Given Observation Sequence
    # unary , pairwise potential are in log space
    N = np.minimum(s, t)
    K = pairwise_potential.shape[0]
    H_alphas = np.zeros((N, K))
    unary_potential = np.copy(unary_potential)
    unary_potential_q = np.copy(unary_potential_q)
    unary_potential[0, :] = unary_potential[0, :] + pairwise_potential[start_tag, :]
    unary_potential[N - 1, :] = unary_potential[N - 1, :] + pairwise_potential[:, end_tag]
    unary_potential_q[0, :] = unary_potential_q[0, :] + pairwise_potential[start_tag, :]
    unary_potential_q[N - 1, :] = unary_potential_q[N - 1, :] + pairwise_potential[:, end_tag]

    C = np.zeros((N, K))
    C_q = np.zeros((N, K))

    C[0, :] = unary_potential[0, :] - logsumexp(unary_potential[0, :])  # logsumexp trick
    C_q[0, :] = unary_potential_q[0, :] - logsumexp(unary_potential_q[0, :])  # logsumexp trick
    for t in range(1, N):
        d = np.zeros(K, dtype=np.float64)
        for k in range(0, K):
            d[k] = np.log(
                np.sum(
                    np.exp(C[t - 1, :] + pairwise_potential[:, k] + unary_potential[t, k])
                    # * np.exp(pairwise_potential[:, k])
                    # * np.exp(unary_potential[t, k])
                )
            )
        d = d - logsumexp(d)
        d_q = np.zeros(K, dtype=np.float64)
        for k in range(0, K):
            d_q[k] = np.log(
                np.sum(np.exp(C_q[t - 1, :] + pairwise_potential[:, k] + unary_potential_q[t, k]))
            )
        d_q = d_q - logsumexp(d_q)
        for j in range(0, K):
            C[t, j] = d[j]
            C_q[t, j] = d_q[j]
            h_next = 0
            logprob = (
                pairwise_potential[:, j]
                + C[t - 1, :]
                - logsumexp(pairwise_potential[:, j] + C[t - 1, :])
            )
            logprob_q = (
                pairwise_potential[:, j]
                + C_q[t - 1, :]
                - logsumexp(pairwise_potential[:, j] + C_q[t - 1, :])
            )
            h_next = (
                H_alphas[t - 1, :] * np.exp(logprob)
            ).sum()  # may be i can convert later part to log space as well
            H_alphas[t, j] = h_next + entropy(np.exp(logprob), np.exp(logprob_q))
    return np.sum(H_alphas[N - 1, :] * np.exp(C[N - 1, :])) + entropy(
        np.exp(C[N - 1, :]), np.exp(C_q[N - 1, :])
    )


def mean_pooling(model_output_last_hidden_state, attention_mask):
    token_embeddings = model_output_last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def compute_entropy_vectorized_torch(
    unary_potential, pairwise_potential, word_seq_len, start_tag, end_tag
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N, K = unary_potential.shape
    H_alphas = torch.zeros((B, N, K), dtype=torch.float64, device=device)
    C = torch.zeros((B, N, K), dtype=torch.float64, device=device)
    unary_potential_vec = unary_potential.clone()
    unary_potential_vec[:, 0, :] = unary_potential_vec[:, 0, :] + pairwise_potential[start_tag, :]
    for i, sen_len in enumerate(word_seq_len):
        unary_potential_vec[i, sen_len - 1, :] = (
            unary_potential_vec[i, sen_len - 1, :] + pairwise_potential[:, end_tag]
        )
    lsp = torch.logsumexp(unary_potential_vec[:, 0, :], dim=1)
    C[:, 0, :] = unary_potential_vec[:, 0, :] - lsp.unsqueeze(1)
    H_alphas = torch.zeros((B, N, K), dtype=torch.float64, device=device)
    for t in range(1, N):
        d = torch.zeros((B, K), dtype=torch.float64, device=device)
        for k in range(0, K):
            aa = (
                C[:, t - 1, :]
                + unary_potential_vec[:, t, k].unsqueeze(1)
                + pairwise_potential[:, k]
            )
            d[:, k] = torch.logsumexp(aa, dim=1)
        d = d - d.logsumexp(dim=1).unsqueeze(1)
        for j in range(0, K):
            C[:, t, j] = d[:, j]
            pp = pairwise_potential
            other = torch.logsumexp(pp[:, j] + C[:, t - 1, :], dim=1).unsqueeze(1)
            log_prob = pp[:, j] + C[:, t - 1, :] - other
            exp_log_prob = log_prob.exp()
            h_next = (H_alphas[:, t - 1, :] * exp_log_prob).nansum(dim=1)
            H_alphas[:, t, j] = h_next + torch.nansum(
                -exp_log_prob.T * exp_log_prob.T.log(), dim=0
            )
    entropy_vec = torch.zeros(B, dtype=torch.float64)
    for i, sen_len in enumerate(word_seq_len):
        entropy_vec[i] = (H_alphas[i, sen_len - 1, :] * C[i, sen_len - 1, :].exp()).nansum() + (
            -C[i, sen_len - 1, :] * C[i, sen_len - 1, :].exp()
        ).nansum()
    return entropy_vec
