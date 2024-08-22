import time

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelWithLMHead


# code from  https://github1s.com/forest-snow/alps/
def kmeans(X, k):
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


def get_mlm_loss(model, inputs):
    """Obtain masked language modeling loss from [model] for tokens in [inputs].

    Should return batch_size X seq_length tensor.
    """
    # fill in the model.
    # zero is loss and one is logits
    with torch.no_grad():
        logits = model(**inputs)[1]
    # zeros is mlm loss and one is the token
    # generate masked lm , can do it using hf dataset process.
    # masked_lm_labels  is depreceted use labels instead
    labels = inputs["labels"]
    batch_size, seq_length, vocab_size = logits.size()
    loss_fct = CrossEntropyLoss(reduction="none")
    loss_batched = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
    loss = loss_batched.view(batch_size, seq_length)
    return loss


def mask_tokens(inputs, tokenizer):
    """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10%
    original."""
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    # args.mlm_probability)
    # probability_matrix = torch.full(labels.shape, args.mlm_probability)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    return inputs, labels


class ALPS:
    """[summary] https://arxiv.org/abs/2010.09535 Cold-start Active Learning through Self-
    supervised Language Modeling."""

    def query(self, model_dir, pool, config, random_initialization=False):
        selection_time = 0.0
        inference_time = 0.0
        if random_initialization:
            NUM_SAMPLES = config.activelearning.initial_labeled
        else:
            NUM_SAMPLES = config.activelearning.query_size
        if isinstance(config.model.config.embedder_type, str):
            model = AutoModelWithLMHead.from_pretrained(config.model.config.embedder_type)
        else:
            raise ValueError(
                "ALPS expects the path of the model or name of the mode eg. bert-base-cased"
            )
        # Use Sequential Sampler  to be consistent with the sampling order as Kmeans
        # is sensitive to dataloader sampling order
        tokenizer = pool.tokenizer
        eval_dataloader = pool.get_unlabeled_data(
            sequential_sample=False,
            batch_size=config.datamodule.eval_batch_size,
        )
        unlabeled_idx = pool.get_unlabeled_indices()
        all_scores_or_vectors = []

        start = time.perf_counter()
        model.eval()
        batch_ind = []

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        for batch in eval_dataloader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            ind = batch["id"]
            batch_ind += ind.cpu()
            # batch = tuple(t.to(args.device) for t in batch)
            # batch = tuple(t.to("cpu") for t in batch)
            inputs = {}
            # mask_tokens() requires CPU input_ids
            # asusme that is for lm
            input_ids_cpu = batch["input_ids"].cpu().clone()
            _, labels = mask_tokens(input_ids_cpu, tokenizer)
            # change the label
            # input_ids = input_ids_mask if args.masked else batch[0]
            input_ids = batch["input_ids"]
            inputs["input_ids"] = input_ids.to(device)
            inputs["labels"] = labels.to(device)
            inputs["attention_mask"] = batch["attention_mask"].to(device)

            with torch.no_grad():
                scores_or_vectors = get_mlm_loss(model=model, inputs=inputs)
                # in case of mlm loss it return score
            for sample in scores_or_vectors:
                all_scores_or_vectors.append(sample.detach().cpu())
            for k, v in batch.items():
                del v
        all_scores_or_vectors = pad_sequence(
            all_scores_or_vectors, batch_first=True, padding_value=0
        )
        # figure out the indices and append the model from checkpoint.
        vectors = normalize(all_scores_or_vectors)
        end = time.perf_counter()
        inference_time += end - start
        #  compute the  centers and return  closest point to the cluster center
        start = time.perf_counter()
        queries_unsampled = kmeans(vectors, k=NUM_SAMPLES)
        queries = torch.LongTensor(queries_unsampled)
        assert len(queries) == len(queries.unique()), "Duplicates found in sampling"
        assert len(queries) > 0, "Sampling method sampled no queries."
        batch_ind = np.array(batch_ind, dtype=np.int64)
        end = time.perf_counter()
        selection_time += end - start
        # we select closest point to centroids based on the surprisal_embedding.
        results = {
            "selection_time": selection_time,
            "inference_time": inference_time,
            "surprisal_embedding": vectors,
            "order": unlabeled_idx[queries].numpy(),
        }
        return results
