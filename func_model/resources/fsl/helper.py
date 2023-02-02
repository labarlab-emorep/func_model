"""Helper methods for FSl-based pipelines."""

import importlib.resources as pkg_resources
from func_model import reference_files


def valid_name(model_name: str) -> bool:
    "Check if model name is valid."
    return model_name in ["sep"]


def valid_level(model_level: str) -> bool:
    "Check if model level is valid."
    return model_level in ["first"]


def valid_task(task: str) -> bool:
    "Check if task name is valid."
    return task in ["task-movies", "task-scenarios"]


def load_reference(file_name: str) -> str:
    """Return FSF template from resources."""
    with pkg_resources.open_text(reference_files, file_name) as tf:
        tp_line = tf.read()
    return tp_line
