import time

import torch


class NormalizedLeastConfidenceStrategy:
    def query(self, model, pool, config):
        """Computes Least Confident samples but normalized by the sentences length."""
        results = dict()
        unlabeleld_dataloader = pool.get_unlabeled_data(
            sequential_sample=False,
            batch_size=config.datamodule.eval_batch_size,
        )
        # both score and z are in log space
        start = time.perf_counter()
        model.eval()
        seq_len = []
        scores = []
        batch_indices = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for batch in unlabeleld_dataloader:
            for k, v in batch.items():
                batch[k] = v.to(device)
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
            log_prob = score - z
            scores.append(log_prob)
            seq_len.append(batch["word_seq_len"])
            batch_indices += list(batch["id"])
        shuffled_indices = torch.tensor(batch_indices).cpu()
        sen_len = torch.cat(seq_len)
        scores = torch.cat(scores)
        normalized_scores = scores - torch.log(sen_len)
        end = time.perf_counter()
        results["inference_time"] = end - start
        results["selection_time"] = 0.0
        shuffled_sorted_indices = shuffled_indices[torch.argsort(normalized_scores).cpu()]
        results["scores"] = normalized_scores.cpu().numpy()
        results["order"] = shuffled_sorted_indices.numpy()
        return results
