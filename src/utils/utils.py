import time
import urllib
from functools import wraps
from pathlib import Path
from typing import Callable, Dict

import mlflow
import numpy as np
import torch
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig


def mlflow_log_runtime(mlflow_run_obj: mlflow.entities.Run):
    """decorator for logging the runtime of a function."""

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            client = MlflowClient()
            start = time.perf_counter()
            out = func(*args, **kwargs)
            end = time.perf_counter()
            if isinstance(mlflow_run_obj, mlflow.entities.Run):
                client.log_metric(
                    mlflow_run_obj.info.run_id, str(func.__name__) + "_runtime", end - start
                )
                return out
            else:
                return {"func_out": out, "runtime": end - start}

        return wrapper

    return decorate


def mlflow_log_dict(run: mlflow.entities.run.Run):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            assert isinstance(out, Dict)
            if not isinstance(out, dict):
                raise ValueError("mlflow_log_dict expect a mapping(dict) with floats to log ")
            client = MlflowClient()
            only_np_array = {k: v for k, v in out.items() if isinstance(v, (np.ndarray))}
            only_torch_tensor = {k: v for k, v in out.items() if isinstance(v, (torch.Tensor))}
            mapping_with_floats = {k: v for k, v in out.items() if isinstance(v, (float, int))}
            artifact_filepath = (
                urllib.parse.urlparse(run.info.artifact_uri).path + "/" + func.__name__
            )
            if only_np_array != {}:
                with open(artifact_filepath + ".npz", "wb") as f:
                    np.savez(f, **only_np_array)
            if only_torch_tensor != {}:
                torch.save(only_torch_tensor, artifact_filepath + ".pt")
            if mapping_with_floats != {}:
                for k, v in mapping_with_floats.items():
                    client.log_metric(run.info.run_id, k, v)
            return out

        return wrapper

    return decorate
