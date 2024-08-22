import time

import torch
from tqdm import tqdm


class LeastConfidence:
    """Computes the log probablitiy of most likely sentence  and the return the indices of least
    confident sentences."""

    def query(self, model, pool, config):
        """
        return:
            results : dict
            sampled_id , which is the id of the samples in the 'id' column of dataset.
        """
        results = {}
        scores = []
        shuffled_indices = []
        # explicit is better than implicit
        unlabeleld_dataloader = pool.get_unlabeled_data(
            sequential_sample=False,
            batch_size=config.datamodule.eval_batch_size,
        )
        start = time.perf_counter()
        model.eval()
        for i, batch in enumerate(tqdm(unlabeleld_dataloader)):
            ind = batch["id"]
            with torch.no_grad():
                score, _, _, _ = model.decode(
                    subword_input_ids=batch["input_ids"].to("cuda"),
                    word_seq_lens=batch["word_seq_len"].to("cuda"),
                    orig_to_tok_index=batch["orig_to_tok_index"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                )
                z = model.compute_Z(
                    subword_input_ids=batch["input_ids"].to("cuda"),
                    word_seq_lens=batch["word_seq_len"].to("cuda"),
                    orig_to_tok_index=batch["orig_to_tok_index"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                    labels=batch["label_ids"].to("cuda"),
                )
                score = score.detach()
                z = model.model.compute_Z(batch).detach()
            log_prob = score - z
            scores.append(log_prob)
            shuffled_indices += ind
        shuffled_indices = torch.IntTensor(shuffled_indices).numpy()
        scores = torch.cat(scores)
        end = time.perf_counter()
        results["inference_time"] = end - start
        results["selection_time"] = 0.0
        results["scores"] = scores
        shuffled_sorted_indices = shuffled_indices[torch.argsort(scores)]
        results["order"] = shuffled_sorted_indices
        return results
