"""Helper methods for FSl-based pipelines.

valid_name : check for valid model name
valid_level : check for valid model level
valid_task : check for valid task name
valid_contrast : check for valid contrast
valid_preproc : check for valid preproc type
load_reference : return reference file content
count_vol : get number of volumes
get_tr : get TR length
load_tsv : read tsv as pd.DataFrame
clean_up : delete unneeded files and copy to group

"""

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
    """Check if model name is valid."""
    return model_name in ["sep", "rest", "lss"]


def valid_level(model_level: str) -> bool:
    """Check if model level is valid."""
    return model_level in ["first"]


def valid_task(task: str) -> bool:
    """Check if task name is valid."""
    return task in ["task-movies", "task-scenarios", "task-rest"]


def valid_contrast(con: str) -> bool:
    """Check if contrast name is valid."""
    return con in ["stim", "replay"]


def valid_preproc(step: str) -> bool:
    """Check if preproc step/type is valid."""
    return step in ["smoothed", "scaled"]


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


def clean_up(subj_work, subj_final, model_name):
    """Remove unneeded files and save rest to group location.

    Parameters
    ----------
    subj_work : str, os.PathLike
        Output work location for intermediates
    subj_final : str, os.PathLike
        Final output location, for storage and transfer
    model_name : str
        Name of model (e.g. 'sep')

    Raises
    ------
    FileNotFoundError
        Session directory not found in subj_final

    """

    def _rm_files(rm_list: list):
        """os.remove files from list of paths."""
        if not rm_list:
            return
        for rm_path in rm_list:
            os.remove(rm_path)

    def _clean_lss():
        """Remove unneeded files from LSS model."""
        # Clean parent model dir
        run_str = f"{subj_work}/run*_name-lss*"
        for rm_par in ["absbrain", "confound", "example"]:
            rm_list = glob.glob(f"{run_str}/{rm_par}*")
            _rm_files(rm_list)
        for rm_dir in ["logs", "custom_timing_files"]:
            rm_path = glob.glob(f"{run_str}/{rm_dir}")
            for _path in rm_path:
                shutil.rmtree(_path)

        # Clean stat dir
        for rm_stat in ["pe", "threshac", "res", "varcope", "sigma"]:
            rm_list = glob.glob(f"{run_str}/stats/{rm_stat}*.nii.gz")
            _rm_files(rm_list)

    # Remove unneeded files
    if model_name == "lss":
        _clean_lss()
    rm_list = glob.glob(
        f"{subj_work}/**/filtered_func_data.nii.gz", recursive=True
    )
    _rm_files(rm_list)

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
