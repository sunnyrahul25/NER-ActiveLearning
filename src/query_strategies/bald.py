import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from src import utils
from src.utils.data_utils import START_TAG, STOP_TAG


class BALD:
    """[summary] Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds
    https://arxiv.org/abs/1906.03671."""

    def query(self, model, pool, config):
        """
        Note: bald implementation is dependent on the batch size
        results would not be reproducible with different batch size
        due to use of dropout layers in the train mode.
        """
        MC_SAMPLES = config.activelearning.mc_samples
        selection_time = 0.0
        inference_time = 0.0
        start = time.perf_counter()
        pairwise_potentials = model.model.inferencer.transition.detach().numpy()
        unlabeled_data = pool.get_unlabeled_data(
            sequential_sample=True,
            batch_size=config.datamodule.eval_batch_size,
        )  # ! sampler must be sequential
        probs_k = []
        seq_len_k = []
        batch_ind_k = []
        model.train()
        for _ in range(0, MC_SAMPLES):
            unlab_model_out = {}
            for dl, out in zip([unlabeled_data], [unlab_model_out]):
                seq_len = []
                probs = []
                batch_ind = []
                for batch in dl:
                    ind = batch["id"]
                    with torch.no_grad():
                        _, _, model_output_last_hidden, _ = model(
                            subword_input_ids=batch["input_ids"].to("cuda"),
                            word_seq_lens=batch["word_seq_len"].to("cuda"),
                            orig_to_tok_index=batch["orig_to_tok_index"].to("cuda"),
                            attention_mask=batch["attention_mask"].to("cuda"),
                        )
                    seq_len += batch["word_seq_len"]
                    probs += model_output_last_hidden
                    batch_ind += ind

                #  is that detach necessary
                out["probs"] = pad_sequence(probs, batch_first=True).detach().numpy()
                out["seq_len"] = [int(x) for x in seq_len]
                out["batch_indices"] = np.array(batch_ind)
            probs_k.append(unlab_model_out["probs"])
            seq_len_k.append(unlab_model_out["seq_len"])
            batch_ind_k.append(unlab_model_out["batch_indices"])

        # average the logits and then compute entropy
        end = time.perf_counter()
        inference_time += end - start
        start = time.perf_counter()
        mean_potential = np.mean(probs_k, axis=0)
        entropy_mean_potential = utils.compute_entropy_vectorized(
            mean_potential,
            pairwise_potentials,
            seq_len_k[0],
            pool.label2idx[START_TAG],
            pool.label2idx[STOP_TAG],
        )
        entropy_of_potentials = []
        for ind, k_potential in enumerate(probs_k):
            entropy_k = utils.compute_entropy_vectorized(
                k_potential,
                pairwise_potentials,
                seq_len_k[ind],
                pool.label2idx[START_TAG],
                pool.label2idx[STOP_TAG],
            )
            entropy_of_potentials.append(entropy_k)
        entropy_of_potentials = np.array(entropy_of_potentials)
        mean_entropy_of_potential = entropy_of_potentials.mean(axis=0)
        bald_score = entropy_mean_potential - mean_entropy_of_potential
        end = time.perf_counter()
        selection_time += end - start
        # select samples with highest bald score.
        results = {
            "selection_time": selection_time,
            "inference_time": inference_time,
            "bald_scores": bald_score,
            "order": batch_ind_k[0][np.argsort(bald_score)[::-1]],
        }
        return results
