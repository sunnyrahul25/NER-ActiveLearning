"""This module implements BADGE algorithm."""
import time

import numpy as np
import torch
from tqdm import tqdm


# code from https://github.com/forest-snow/alps
def closest_center_dist(X, centers):
    """Compute distance to nearest centers.

    :param X : matrix of vectors
    :param centers : indices of centers
    """
    dist = torch.cdist(X, X[centers])
    cd = dist.min(axis=1).values
    return cd


def kmeans_pp(vectors, k, centers):
    """Perform kmean clusters on vectors.

    :param vectors [TODO:type]: [TODO:description]
    :param k [TODO:type]: number of clusters
    :param centers [TODO:type]: [TODO:description]
    """
    if vectors.size(0) == k:  # dont perform k means if number of centers is same as samples.
        return np.arange(k)
    if len(centers) == 0:
        # randomly choose first center
        c1 = np.random.choice(vectors.size(0))
        centers.append(c1)
        k -= 1
    # greedily choose centers
    for _ in tqdm(range(k)):
        dist = closest_center_dist(vectors, centers) ** 2
        prob = (dist / dist.sum()).cpu().detach().numpy()
        ci = np.random.choice(vectors.size(0), p=prob)
        centers.append(ci)
    return centers


# BADGE algorithm (Ash et al. 2020)
def badge(grads, k):
    """Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds.

    (BADGE)  Ash. et al https://arxiv.org/abs/1906.03671
    :param grads :
    :param k :
    """
    centers = kmeans_pp(grads, k, [])
    return centers


class BADGE:
    """"""

    def query(self, model, pool, config):
        """Compute samples to be annotated given model and pool config.

        :param model [TODO:type]: [TODO:description]
        :param pool [TODO:type]: [TODO:description]
        :param config [TODO:type]: [TODO:description]
        """
        NUM_SAMPLES = config.activelearning.query_size
        EMB_TYPE = "cls"
        selection_time = 0.0
        inference_time = 0.0

        start = time.perf_counter()
        unlab_model_out = {}
        unlabeled_data_loader = pool.get_unlabeled_data(
            sequential_sample=False,
            batch_size=config.datamodule.eval_batch_size,
        )
        batch_ind = []
        emb = []
        model.eval()
        for batch in unlabeled_data_loader:
            ind = batch["id"]
            # set batch label  to model prediction
            with torch.no_grad():
                _, best_path, _, _ = model.decode(
                    subword_input_ids=batch["input_ids"].to("cuda"),
                    word_seq_lens=batch["word_seq_len"].to("cuda"),
                    orig_to_tok_index=batch["orig_to_tok_index"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                )
            # compute the loss and gradient embedding
            _, gradient_emb = model.compute_gradient(
                subword_input_ids=batch["input_ids"].to("cuda"),
                word_seq_lens=batch["word_seq_len"].to("cuda"),
                orig_to_tok_index=batch["orig_to_tok_index"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                labels=best_path,
            )
            if EMB_TYPE == "cls":
                cls_grad_emb = gradient_emb[:, 0, :]  # 0 CLS Token
                emb.append(cls_grad_emb)
            batch_ind += ind

        unlab_model_out["embeddings"] = torch.cat(emb)
        unlab_model_out["batch_indices"] = np.array(batch_ind)
        end = time.perf_counter()
        inference_time += end - start
        start = time.perf_counter()
        centers = badge(unlab_model_out["embeddings"], NUM_SAMPLES)
        end = time.perf_counter()
        selection_time += end - start
        return {
            "gradient_embedding": unlab_model_out["embeddings"],
            "order": unlab_model_out["batch_indices"][centers],
        }
