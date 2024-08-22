import numpy as np


class RandomSampling:
    def query(self, model, pool, config):
        """[summary]"""
        tmp_idxs = np.copy(pool.get_unlabeled_indices())
        np.random.shuffle(tmp_idxs)
        return {
            "order": tmp_idxs,
            "selection_time": 0.0,
            "inference_time": 0.0,
        }
