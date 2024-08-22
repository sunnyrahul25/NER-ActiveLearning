import numpy as np


class MinimumTokenProbabilityStrategy:
    def query(self, num_samples, scores, best_path, probs, *args, **kwargs):
        assert probs.shape[0] == scores.shape[0] == len(best_path)
        mtp_socres = []
        for prob, path in zip(probs, best_path):
            prob = prob[: len(path)]
            prob -= np.max(prob)
            prob = np.exp(prob) / np.sum(np.exp(prob))
            mtp_socres.append(np.min(np.max(prob[: len(path)], axis=1)))
        idx = np.argpartition(mtp_socres, range(num_samples))[:num_samples]
        return idx
