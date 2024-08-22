import time

import torch


class LeastTokenProbability:
    # code from https://github.com/HIT-ICES/AL-NER/blob/master/modules/select_strategy/ALStrategy.py
    def query(self, model, pool, config):
        results = dict()
        batch_ind = []
        dl = pool.get_unlabeled_data(
            sequential_sample=False,
            batch_size=config.datamodule.eval_batch_size,
        )
        ltp_scores = []
        inference_time = 0.0
        selection_time = 0.0
        model.eval()
        for batch in dl:
            ind = batch["id"]
            start = time.perf_counter()
            with torch.no_grad():
                _, best_path, prob, _ = model(
                    subword_input_ids=batch["input_ids"].to("cuda"),
                    word_seq_lens=batch["word_seq_len"].to("cuda"),
                    orig_to_tok_index=batch["orig_to_tok_index"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                )

            end = time.perf_counter()
            inference_time += end - start

            start = time.perf_counter()
            for pr, path, w_len in zip(prob, best_path, batch["word_seq_len"]):
                ltp_scores.append(torch.min(torch.take(pr, path)[:w_len]))
            batch_ind += ind
            end = time.perf_counter()
            selection_time += end - start

        batch_ind = torch.IntTensor(batch_ind).numpy()
        scores = torch.FloatTensor(ltp_scores)
        end = time.perf_counter()

        results["inference_time"] = inference_time
        results["selection_time"] = selection_time
        results["ltp_scores"] = scores
        results["order"] = batch_ind[torch.argsort(scores)]
        return results
