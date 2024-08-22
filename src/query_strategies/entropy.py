import time

import numpy as np
import torch

from src import utils
from src.utils.data_utils import START_TAG, STOP_TAG


class Entropy:
    """[summary]"""

    def query(self, model, pool, config):
        """Computes entropy for a sequence and return indices for sequences with maximum
        entropy."""
        inference_time = 0.0
        selection_time = 0.0
        unlabeled_dataloader = pool.get_unlabeled_data(
            sequential_sample=False,
            batch_size=config.datamodule.eval_batch_size,
        )
        model.eval()
        seq_len = []
        batch_ind = []
        entropy_l = []
        pairwise_potentials = model.inferencer.transition.detach()
        for batch in unlabeled_dataloader:
            start = time.perf_counter()
            with torch.no_grad():
                _, _, model_output_last_hidden, _ = model.decode(
                    subword_input_ids=batch["input_ids"].to("cuda"),
                    word_seq_lens=batch["word_seq_len"].to("cuda"),
                    orig_to_tok_index=batch["orig_to_tok_index"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                )
            seq_len += batch["word_seq_len"]
            batch_ind.append(batch["id"])
            end = time.perf_counter()
            inference_time += end - start
            start = time.perf_counter()
            entropy = utils.compute_entropy_vectorized(
                model_output_last_hidden.cpu().numpy(),
                pairwise_potentials.cpu().numpy(),
                batch["word_seq_len"].cpu().numpy(),
                pool.label2idx[START_TAG],
                pool.label2idx[STOP_TAG],
            )
            entropy_l.append(entropy)
            end = time.perf_counter()
            selection_time += end - start

        start = time.perf_counter()
        entropy_scores = np.concatenate(entropy_l)
        batch_ind = torch.cat(batch_ind).cpu().numpy()
        # sorted in reverse order because we want to select samples with
        # maximum entropy
        sorted_ind = batch_ind[np.argsort(entropy_scores)[::-1]]
        sorted_scores = np.sort(entropy_scores)[::-1]
        end = time.perf_counter()
        selection_time += end - start

        return {
            "inference_time": inference_time,
            "selection_time": selection_time,
            "scores": sorted_scores,
            "order": sorted_ind,
        }
