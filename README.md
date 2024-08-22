# NER Active Learning
This repository contains the implementation and empirical evaluation of various active learning strategies applied to the Named Entity Recognition (NER) task. The following strategies have been implemented and evaluated:
- Neural Least Confidence (NLC)
- Entropy-Based Sampling (Entropy)
- Contrastive Active Learning (Contrastive)
- Batch Active Learning by Diverse Gradient Embeddings (BADGE)
- Active Learning by Processing Surprisals (ALPS)

## How to run
Install dependencies

```bash
# clone project
git clone https://github.com/y1450/NER-ActiveLearning
cd NER-ActiveLearning

# [OPTIONAL] create conda environment
conda create -n neral python=3.9
conda activate neral

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train_al.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

## Running on custom dataset

The preprocessing of data is kept separate from the codebase,
the code expects that you create a huggingface dataset with
following specification

dataset must contain columns named "ner_tags" and "tokens".

overall dataset dict have the following structure
`dataset [split]`
for split there must be `pool`,`validation`,`test`

The tokenization is handled by the code and your dataset is
expected to in text.
eg tokens:'Hellow World','ner_tags':'0 0 1 3'

The tag mapping is used from the train split of dataset.

Example

```python
# create a small subset of 100 conll2003 samples
from datasets import load_dataset
conll2003 = load_dataset("conll2003")
conll2003["pool"]=conll2003["train"].shuffle(seed=42).select(range(100))
conll2003["test"]=conll2003["test"].shuffle(seed=42).select(range(100))
conll2003["validation"]=conll2003["validation"].shuffle(seed=42).select(range(100))
print(conll2003)
#>>> print(conll2003)
#DatasetDict({
#    train: Dataset({
#        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
#        num_rows: 14041
#    })
#    validation: Dataset({
#        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
#        num_rows: 100
#    })
#    test: Dataset({
#        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
#        num_rows: 100
#    })
#    pool: Dataset({
#        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
#        num_rows: 100
#    })
#})
#>>>
```

set the data_seed in the experiment config

### To extend a new strategy

Create class and implement the query method

the `query` method  has the following signature `def query(self, model, pool, config, num_samples)`
and must return dict with key `order`. the `order` is `id` of pool samples that model
wants to annotate, in ascending order.

if `query size` is `k`, then first k element of order will be labeled.

You can also return any thing that might be useful for logging eg. scores,
runtime.

```python
class LeastConfidence:
    """Computes the log probablitiy of most likely sentence  and the return
    the indices of least confident sentences
    """
    def query(self, model, pool, config, num_samples):
      pass
```

### Running Experiment

`docker build -f ./docker/Dockerfile --build-arg USER_UID=$UID --build-arg USER_NAME=$(id -un) -t docker.lsv.uni-saarland.de/srahul/ner-al-22.02-py3 .`
