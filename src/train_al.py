"""Active Learning Training Pipeline implementation."""
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import copy
import gc
import logging
import os
import time
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import hydra
import mlflow
import torch
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #
from seqeval.metrics import classification_report
from termcolor import colored
from torch import Tensor
from torch.utils.data import DataLoader

# my imports
from transformers import set_seed

from src import utils
from src.datamodules.al_dataset import ActiveLearningDatasetPool
from src.models.modules.transformers_neuralcrf import TransformersCRF
from utils.transformers_util import get_huggingface_optimizer_and_scheduler

log = logging.getLogger(__name__)


def compute_f1_scores(
    config: Dict, model: TransformersCRF, data_loader: DataLoader
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Computes f1-score using seqeval package."""
    # evaluation
    all_prediction = []
    all_target = []
    with torch.no_grad():
        for batch_id, batch in enumerate(data_loader, 0):
            _, batch_max_ids, _, _ = model.decode(
                subword_input_ids=batch["input_ids"].to("cuda"),
                word_seq_lens=batch["word_seq_len"].to("cuda"),
                orig_to_tok_index=batch["orig_to_tok_index"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
            )
            word_seq_lens = batch["word_seq_len"].tolist()
            for idx in range(len(batch_max_ids)):
                length = word_seq_lens[idx]
                prediction = batch_max_ids[idx][:length].tolist()
                output = batch["label_ids"][idx][:length].tolist()
                output = [config["idx2labels"][i] for i in output]
                prediction = [config["idx2labels"][i] for i in prediction]
                # fix the input
                all_prediction.append(prediction)
                all_target.append(output)
            batch_id += 1
        results = dict()
        results = classification_report(all_target, all_prediction, output_dict=True)
        precision = {k: v["precision"] for k, v in results.items()}  # type: ignore
        recall = {k: v["recall"] for k, v in results.items()}  # type: ignore
        f1_scores = {k: v["f1-score"] for k, v in results.items()}  # type:ignore

    return precision, recall, f1_scores


def evaluate_loss(model: TransformersCRF, data_loader: DataLoader) -> float:
    """Computes model loss the given dataloader."""
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(data_loader, 0):
            loss = model(
                subword_input_ids=batch["input_ids"].to("cuda"),
                word_seq_lens=batch["word_seq_len"].to("cuda"),
                orig_to_tok_index=batch["orig_to_tok_index"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                labels=batch["label_ids"].to("cuda"),
            )
            total_loss += loss.item()
    return total_loss


def train_model(
    model: TransformersCRF,
    config: Dict,
    mlflow_run: Run,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> Dict[str, float]:
    assert 0 < config["val_check_interval"] <= 1
    func_start = time.time()
    train_num = len(train_loader)
    client = MlflowClient()
    log.info("[Data Info] number of training batches: %d", train_num)
    epoch = config["num_epochs"]
    optimizer, scheduler = get_huggingface_optimizer_and_scheduler(
        config,
        model,
        num_training_steps=len(train_loader) * epoch,
        weight_decay=config["weight_decay"],
        eps=config["adam_epsilon"],
        warmup_step=config["warmup_steps"],
    )
    log.info(colored("[Optimizer Info] Using the optimizer from pytorch info as you need.", "red"))
    log.info(optimizer)
    best_dev = [-1, 0]
    num_increase_dev = 0
    valid_f1 = 0

    train_time = 0.0
    f1_time = 0.0
    loss_eval_time = 0.0
    STOP_TRAINING = False
    best_model_weights = copy.deepcopy(model.state_dict())
    eval_step = 0

    val_check_after_batch = max(1, int(len(train_loader) * config["val_check_interval"]))
    for i in range(1, epoch + 1):
        client.log_metric(mlflow_run.info.run_id, "epoch", i)
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad(set_to_none=True)
        model.train()
        for batch_i, batch in enumerate(train_loader, 1):
            start = time.time()
            optimizer.zero_grad(set_to_none=True)
            loss = model(
                subword_input_ids=batch["input_ids"].to("cuda"),
                word_seq_lens=batch["word_seq_len"].to("cuda"),
                orig_to_tok_index=batch["orig_to_tok_index"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                labels=batch["label_ids"].to("cuda"),
            )
            epoch_loss += loss.item()
            client.log_metric(mlflow_run.info.run_id, "train/batch_loss", loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip_val"]
            )  # pyright : ignore
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            model.zero_grad(set_to_none=True)
            end = time.time()
            train_time = train_time + end - start
            # Model Evaluation
            if (batch_i % val_check_after_batch) == 0:
                model.eval()
                eval_step += 1
                loss_eval_time = loss_eval_time + end - start
                v_precision, v_recall, v_f1 = compute_f1_scores(config, model, val_loader)
                for k in v_precision.keys():
                    client.log_metric(
                        mlflow_run.info.run_id, "v_precision_" + k, v_precision[k], step=eval_step
                    )
                    client.log_metric(
                        mlflow_run.info.run_id, "v_recall_" + k, v_recall[k], step=eval_step
                    )
                    client.log_metric(mlflow_run.info.run_id, "v_f1_" + k, v_f1[k], step=eval_step)
                valid_f1 = v_f1["micro avg"]
                if valid_f1 > best_dev[0]:
                    num_increase_dev = 0
                    best_dev[0] = valid_f1
                    best_dev[1] = i
                    best_model_weights = copy.deepcopy(model.state_dict())
                else:
                    num_increase_dev += 1

                model.zero_grad(set_to_none=True)
                if num_increase_dev >= config["max_no_incre"]:
                    log.info(
                        "early stop because there are %d steps  valid f1 micro is not increasing on dev"
                        % num_increase_dev
                    )
                    STOP_TRAINING = True
                    break
        end_time = time.time()
        log.info(
            "Epoch %d: Train Loss %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time)
        )
        valid_loss = evaluate_loss(model, val_loader)
        client.log_metric(mlflow_run.info.run_id, "train/epoch_loss", epoch_loss, step=i)
        client.log_metric(mlflow_run.info.run_id, "valid/epoch_loss", valid_loss, step=i)

        if STOP_TRAINING:
            break
    model.load_state_dict(best_model_weights)
    model.eval()
    # Evaluate F1 scores
    start = time.time()
    dev_precision, dev_recall, dev_f1 = compute_f1_scores(config, model, val_loader)
    test_precision, test_recall, test_f1 = compute_f1_scores(config, model, test_loader)
    end = time.time()
    f1_time = f1_time + end - start
    for k in dev_precision.keys():
        client.log_metric(mlflow_run.info.run_id, "dev_precision_" + k, dev_precision[k])
        client.log_metric(mlflow_run.info.run_id, "dev_recall_" + k, dev_recall[k])
        client.log_metric(mlflow_run.info.run_id, "dev_f1_" + k, dev_f1[k])
        client.log_metric(mlflow_run.info.run_id, "test_precision_" + k, test_precision[k])
        client.log_metric(mlflow_run.info.run_id, "test_recall_" + k, test_recall[k])
        client.log_metric(mlflow_run.info.run_id, "test_f1_" + k, test_f1[k])
    func_end = time.time()
    return {
        "train_model_func_runtime": func_end - func_start,
        "f1_runtime": f1_time,
        "loss_eval_time": loss_eval_time,
        "train_runtime": train_time,
    }


def round_run(
    pool: ActiveLearningDatasetPool, config: DictConfig, mlflow_run_obj, round=-1
) -> None:
    """Runs a single iteration of active learning loop."""
    client = MlflowClient()
    if __debug__:
        log.info(f"Current Memory Usage <{torch.cuda.memory_allocated()}>")

    log.info(f"Round <{round}>")
    log.info("Starting training!")
    labeled_dl = pool.get_labeled_data()
    val_dl = pool.get_dataloader(split="validation", batch_size=config.datamodule.eval_batch_size)
    test_dl = pool.get_dataloader(split="test", batch_size=config.datamodule.eval_batch_size)
    num_train_steps = len(pool.get_labeled_data()) * config.trainer.max_epochs

    client.log_param(
        mlflow_run_obj.info.run_id,
        "max_memory_allocated_mb",
        torch.cuda.max_memory_allocated() / (1024 * 1024),
    )
    client.log_param(mlflow_run_obj.info.run_id, "samples", len(pool.get_labeled_indices()))
    client.log_param(
        mlflow_run_obj.info.run_id, "remaining_samples", len(pool.get_unlabeled_indices())
    )
    client.log_param(
        mlflow_run_obj.info.run_id, "activelearning.strategy", config.activelearning.strategy
    )

    model_config = {
        "label_size": len(pool.label2idx),
        "label2idx": pool.label2idx,
        "idx2labels": pool.idx2labels,
        "embedder_type": config.model.config.embedder_type,
        "hidden_dim": config.model.config.hidden_dim,
        "add_iobes_constraint": config.model.config.add_iobes_constraint,
        "dropout": config.model.config.dropout,
        "learningrate": config.model.config.learningrate,
        "adam_epsilon": config.model.config.adam_epsilon,
        "weight_decay": config.model.config.weight_decay,
        "num_epochs": config.trainer.max_epochs,
        "warmup_steps": min(int(0.1 * num_train_steps), 100),
        "max_no_incre": config.callbacks.early_stopping.patience,
        "gradient_clip_val": config.trainer.gradient_clip_val,
        "eval_batch_size": config.datamodule.eval_batch_size,
        "val_check_interval": config.trainer.val_check_interval,
    }

    model = TransformersCRF(model_config)
    model.to("cuda")

    train_model(model, model_config, mlflow_run_obj, labeled_dl, val_dl, test_dl)

    if len(pool.get_unlabeled_indices()) >= config.activelearning.query_size:
        strategy = utils.get_strategy(config.activelearning.strategy)
        out = utils.mlflow_log_dict(mlflow_run_obj)(strategy.query)(model, pool, config)
        sampled_id = out["order"][: config.activelearning.query_size]
        # Assert samples indices do not come from labeled pool
        assert bool(not set(pool.get_labeled_indices()) & set(sampled_id))
        pool.update_idx(Tensor(sampled_id))  # Assert no common data in Dlab and Dpool
        assert bool(not set(pool.get_labeled_indices()) & set(pool.get_unlabeled_indices()))
        # Assert unique (no duplicate) inds in Dlab & Dpool
        assert len(set(pool.get_labeled_indices())) == len(pool.get_labeled_indices())
        assert len(set(pool.get_unlabeled_indices())) == len(pool.get_unlabeled_indices())

    log.info(f"Max Memory Usage <{torch.cuda.max_memory_allocated()}>")
    client.set_terminated(mlflow_run_obj.info.run_id)


def activelearning_pipeline(config: DictConfig):
    """Main active learning loop."""

    if config.get("seed"):
        set_seed(config.seed)

    # _________________ Initialize pool__________________
    client = MlflowClient()
    experiment = mlflow.set_experiment(config.task_name)
    parent_run = client.create_run(experiment.experiment_id)
    artifact_path = urlparse(parent_run.info.artifact_uri).path
    OmegaConf.save(config, str(artifact_path) + "/config.yaml")
    pool = ActiveLearningDatasetPool(config)
    torch.save(pool.initial_pool_indices(), str(artifact_path) + "/init_pool_indices.pt")
    # warm start vs cold start initialization
    if config.activelearning.strategy == "alps":
        strategy = utils.get_strategy(config.activelearning.strategy)
        out = strategy.query(
            config.model.config.embedder_type, pool, config, random_initialization=True
        )
        sampled_id = out["order"][: config.activelearning.initial_labeled]
        # Assert samples indices do not come from labeled pool
        assert bool(not set(pool.get_labeled_indices()) & set(sampled_id))
        pool.update_idx(Tensor(sampled_id))
    else:
        # random initialization
        pool.initialize_labels(config.activelearning.initial_labeled)
    torch.save(
        pool.get_labeled_indices(), str(artifact_path) + "/intial_labeled_samples-warmstart.pt"
    )
    client.log_param(parent_run.info.run_id, "HOSTNAME", os.environ["HOSTNAME"])
    client.log_param(parent_run.info.run_id, "seed", config.seed)
    # ____________ Main Al Loop_____________________
    for round in range(0, config.activelearning.num_rounds + 1):
        mlflow_run_obj = client.create_run(
            experiment.experiment_id, tags={"mlflow.parentRunId": parent_run.info.run_id}
        )
        utils.mlflow_log_runtime(mlflow_run_obj)(round_run)(pool, config, mlflow_run_obj, round)
        gc.collect()
        torch.cuda.empty_cache()
    client.set_terminated(parent_run.info.run_id)
    return 0


@hydra.main(
    version_base="1.2",
    config_path=root / "configs",
    config_name="train.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    return activelearning_pipeline(cfg)


if __name__ == "__main__":
    main()
