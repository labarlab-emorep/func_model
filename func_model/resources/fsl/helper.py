"""Helper methods for FSl-based pipelines."""

import os
import glob
import subprocess
import shutil
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


def clean_up(subj_work, subj_final):
    """Remove unneeded files and save rest to group location.

    Parameters
    ----------
    subj_work : path
        Output work location for intermediates
    subj_final : path
        Final output location, for storage and transfer

    Raises
    ------
    FileNotFoundError
        Session directory not found in subj_final

    """
    # Remove unneeded files
    rm_list = glob.glob(
        f"{subj_work}/**/filtered_func_data.nii.gz", recursive=True
    )
    if rm_list:
        for rm_path in rm_list:
            os.remove(rm_path)

    # Copy remaining files to group location, clean work location
    h_sp = subprocess.Popen(
        f"cp -r {subj_work} {subj_final}", shell=True, stdout=subprocess.PIPE
    )
    _ = h_sp.communicate()
    h_sp.wait()
    chk_save = os.path.join(subj_final, os.path.basename(subj_work))
    if not os.path.exists(chk_save):
        raise FileNotFoundError(f"Expected to find {chk_save}")
    shutil.rmtree(os.path.dirname(subj_work))
