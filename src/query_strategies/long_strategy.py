import time

from torch import IntTensor, argsort
import numpy as np


class LongStrategy:
    def query(self, model, pool, config):
        seq_len = []
        batch_ind = []
        unlabeleld_dataloader = pool.get_unlabeled_data(
            sequential_sample=False,
            batch_size=config.datamodule.eval_batch_size,
        )

        start = time.perf_counter()
        for batch in unlabeleld_dataloader:
            ind = batch["id"]
            seq_len += batch["word_seq_len"]
            batch_ind += ind
        batch_ind = IntTensor(batch_ind)
        seq_len = IntTensor(seq_len)
        end = time.perf_counter()
        idx = argsort(seq_len, descending=True)

        return {
            "order": batch_ind[idx].numpy(),
            "selection_time": 0.0,
            "inference_time": end - start,
        }
