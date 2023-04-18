"""Helper methods for FSl-based pipelines."""

import os
import glob
import subprocess
import shutil
from typing import Union
import pandas as pd
import nibabel as nib
import importlib.resources as pkg_resources
from func_model import reference_files


def valid_name(model_name: str) -> bool:
    "Check if model name is valid."
    return model_name in ["sep", "rest"]


def valid_level(model_level: str) -> bool:
    "Check if model level is valid."
    return model_level in ["first"]


def valid_task(task: str) -> bool:
    "Check if task name is valid."
    return task in ["task-movies", "task-scenarios", "task-rest"]


def valid_contrast(con: str) -> bool:
    "Check if contrast name is valid."
    return con in ["stim", "replay"]


def load_reference(file_name: str) -> str:
    """Return FSF template from resources."""
    with pkg_resources.open_text(reference_files, file_name) as tf:
        tp_line = tf.read()
    return tp_line


def count_vol(in_epi: Union[str, os.PathLike]) -> int:
    """Return number of EPI volumes."""
    img = nib.load(in_epi)
    img_header = img.header
    num_vol = img_header.get_data_shape()[3]
    return num_vol


def get_tr(in_epi: Union[str, os.PathLike]) -> float:
    """Return TR length."""
    img = nib.load(in_epi)
    img_header = img.header
    len_tr = img_header.get_zooms()[3]
    return len_tr


def load_tsv(tsv_path: Union[str, os.PathLike]) -> pd.DataFrame:
    print(f"\t\tLoading {tsv_path} ...")
    return pd.read_csv(tsv_path, sep="\t")


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
