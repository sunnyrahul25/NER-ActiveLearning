import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from src.utils.active_learning import mean_pooling


# cod from https://github.com/forest-snow/alps
def closest_center_dist(X, centers):
    # return distance to closest center
    dist = torch.cdist(X, X[centers])
    cd = dist.min(axis=1).values
    return cd


def kmeans(X, k, tol=1e-4, **kwargs):
    # kmeans algorithm
    print("Running Kmeans")
    kmeans = KMeans(n_clusters=k).fit(X)
    centers = kmeans.cluster_centers_
    # find closest point to centers
    centroids = cdist(centers, X).argmin(axis=1)
    centroids_set = np.unique(centroids)
    m = k - len(centroids_set)
    if m > 0:
        pool = np.delete(np.arange(len(X)), centroids_set)
        p = np.random.choice(len(pool), m)
        centroids = np.concatenate((centroids_set, pool[p]), axis=None)
    return centroids


class BERTKM:
    """Kmeans clustering on L2 normalized bert embeddings."""

    def query(self, model, pool, config, num_samples):
        NUM_SAMPLES = config.activelearning.query_size
        EMB_TYPE = "cls"
        unlab_model_out = {}
        unlabeled_idx = pool.get_unlabeled_indices()
        unlabeled_data_loader = pool.get_unlabeled_data(sequential_sample=True)
        batch_ind = []
        emb = []
        for batch, ind in unlabeled_data_loader:
            model.eval()
            # set batch label  to model prediction
            with torch.no_grad():
                score, best_path, model_output_last_hidden, _ = model(
                    subword_input_ids=batch["input_ids"].to("cuda"),
                    word_seq_lens=batch["word_seq_len"].to("cuda"),
                    orig_to_tok_index=batch["orig_to_tok_index"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                )
            if EMB_TYPE == "cls":
                cls_emb = model_output_last_hidden[:, 0, :]  # 0 CLS Token
                emb.append(cls_emb)
            if EMB_TYPE == "pool":
                pool_emb = mean_pooling(model_output_last_hidden, batch.attention_mask)
                emb.append(pool_emb)
            batch_ind += ind

        unlab_model_out["embeddings"] = torch.cat(emb)
        unlab_model_out["batch_indices"] = np.array(batch_ind)
        centers = kmeans(unlab_model_out["embeddings"], NUM_SAMPLES)
        return unlab_model_out["embeddings"], unlabeled_idx[centers]
