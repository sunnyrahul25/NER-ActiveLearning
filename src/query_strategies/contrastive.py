import time

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src import utils
from src.utils.data_utils import START_TAG, STOP_TAG


class Contrastive:
    """[summary]"""

    # based on  Efficient Computation of the Hidden Markov Model Entropy for a Given Observation Sequence
    def query(self, model, pool, config):
        """[summary]

        Returns:
            dict {
            "scores:", contristive scores (higher, more divergent )
            "order:", id of the samples
            }
        """

        assert config.activelearning.neighbors is not None
        NUM_NEIGHBORS = config.activelearning.neighbors
        EMB_TYPE = config.activelearning.emb_type
        inference_time = 0.0
        selection_time = 0.0
        start = time.perf_counter()
        lab_model_out = {}
        unlab_model_out = {}
        labeled_data = pool.get_labeled_data()
        unlabeled_data = pool.get_unlabeled_data(
            sequential_sample=False,
            batch_size=config.datamodule.eval_batch_size,
        )
        model.eval()
        for dl, out in zip(
            [unlabeled_data, labeled_data],
            [unlab_model_out, lab_model_out],
        ):
            scores = []
            best_paths = []
            seq_len = []
            probs = []
            batch_ind = []
            emb = []
            for batch in dl:
                ind = batch["id"]
                with torch.no_grad():
                    score, best_path, model_output_last_hidden, _ = model.decode(
                        subword_input_ids=batch["input_ids"].to("cuda"),
                        word_seq_lens=batch["word_seq_len"].to("cuda"),
                        orig_to_tok_index=batch["orig_to_tok_index"].to("cuda"),
                        attention_mask=batch["attention_mask"].to("cuda"),
                    )

                if EMB_TYPE == "cls":
                    cls_emb = model_output_last_hidden[:, 0, :]  # 0 CLS Token
                    emb.append(cls_emb)
                scores.append(score)
                best_paths += best_path
                seq_len += batch["word_seq_len"]
                probs += model_output_last_hidden
                batch_ind += ind
            assert len(labeled_data) > 0
            out["scores"] = torch.cat(scores)
            out["embeddings"] = torch.cat(emb)
            out["probs"] = torch.nn.utils.rnn.pad_sequence(probs, batch_first=True)
            out["best_paths"] = torch.nn.utils.rnn.pad_sequence(best_paths, batch_first=True)
            out["seq_len"] = [int(x) for x in seq_len]
            out["batch_indices"] = np.array(batch_ind, dtype=np.int32)
        end = time.perf_counter()
        inference_time += end - start
        start = time.perf_counter()
        probs_u = unlab_model_out["probs"].detach().cpu().numpy()
        probs_l = lab_model_out["probs"].detach().cpu().numpy()
        pairwise_potentials = model.inferencer.transition.detach().cpu().numpy()
        sen_len_u = unlab_model_out["seq_len"]
        sen_len_l = lab_model_out["seq_len"]
        embedding_u = unlab_model_out["embeddings"].detach().cpu().numpy()  # unlabeled embedding
        embedding_l = lab_model_out["embeddings"].detach().cpu().numpy()  # labeled embedding
        embedding_u = normalize(embedding_u, axis=1)
        embedding_l = normalize(embedding_l, axis=1)
        neigh = NearestNeighbors(n_neighbors=NUM_NEIGHBORS)
        neigh.fit(embedding_l)  # fit on labeled sentences

        # find k nearest in the unlabeled samples
        _, neighbors = neigh.kneighbors(embedding_u, return_distance=True)

        kl_score_avg = []
        for pool_idx in range(len(probs_u)):
            # filter probs of neighbors
            q = probs_l[neighbors[pool_idx]]  # neighbors potential in labeled space
            sen_p = np.tile(sen_len_u[pool_idx], NUM_NEIGHBORS)
            sen_q = np.array(sen_len_l)[neighbors[pool_idx]]
            pot_p = np.tile(probs_u[pool_idx], (NUM_NEIGHBORS, 1, 1))
            differ = abs(pot_p.shape[1] - q.shape[1])
            if pot_p.shape[1] > q.shape[1]:
                q = np.pad(q, ((0, 0), (0, differ), (0, 0)), mode="constant", constant_values=0)
            else:
                pot_p = np.pad(
                    pot_p, ((0, 0), (0, differ), (0, 0)), mode="constant", constant_values=0
                )
            kl_score = utils.compute_kl_log_vectorized(
                pot_p,
                q,
                pairwise_potentials,
                sen_p,
                sen_q,
                pool.label2idx[START_TAG],
                pool.label2idx[STOP_TAG],
            )
            kl_score_avg.append(kl_score.mean())
        end = time.perf_counter()
        selection_time += end - start
        return {
            "selection_time": selection_time,
            "inference_time": inference_time,
            "scores": np.sort(kl_score_avg)[
                ::-1
            ],  # we select samples with highest kl_divergence with it neighbours
            "order": unlab_model_out["batch_indices"][np.argsort(kl_score_avg)[::-1]],
        }
