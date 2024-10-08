"""Various helper methods.

prepend_afni_sing : setup singularity call for afni
valid_task : check task name
valid_models : check model name
valid_univ_test : check univariate test type
valid_mvm_test : check multivariate test type
valid_name : check for valid model name
valid_level : check for valid model level
valid_contrast : check for valid contrast
valid_preproc : check for valid preproc type
load_reference : return reference file content
count_vol : get number of volumes
get_tr : get TR length
load_tsv : read tsv as pd.DataFrame
clean_up : delete unneeded files and copy to group
emo_switch : supply emotion names
SupportFsl : helper methods for fsl modeling
SyncGroup : download all model_afni data

"""

import os
import glob
import subprocess
import shutil
from typing import Union
from typing import Tuple
import pandas as pd
import nibabel as nib
import importlib.resources as pkg_resources
from func_model import reference_files


def prepend_afni_sing(
    work_deriv: Union[str, os.PathLike], subj_work: Union[str, os.PathLike]
) -> list:
    """Supply singularity call for AFNI."""
    try:
        sing_afni = os.environ["SING_AFNI"]
    except KeyError as e:
        print("Missing required variable SING_AFNI")
        raise e

    return [
        "singularity run",
        "--cleanenv",
        f"--bind {work_deriv}:{work_deriv}",
        f"--bind {subj_work}:{subj_work}",
        f"--bind {subj_work}:/opt/home",
        sing_afni,
    ]


def valid_models(model_name: str) -> bool:
    """Return bool of whether model_name is supported."""
    return model_name in ["rest", "mixed", "task", "block"]


def valid_univ_test(test_name: str) -> bool:
    """Return bool of whether test_name is supported."""
    return test_name in ["student", "paired"]


def valid_mvm_test(test_name: str) -> bool:
    """Return bool of whether test_name is supported."""
    return test_name in ["amodal"]


def valid_name(model_name: str) -> bool:
    """Check if model name is valid."""
    return model_name in ["sep", "tog", "rest", "lss"]


def valid_level(model_level: str) -> bool:
    """Check if model level is valid."""
    return model_level in ["first", "second"]


def valid_task(task: str) -> bool:
    """Check if task name is valid."""
    return task in ["task-movies", "task-scenarios", "task-rest", "task-all"]


def valid_contrast(con: str) -> bool:
    """Check if contrast name is valid."""
    return con in ["stim", "replay", "tog"]


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


def load_csv(csv_path: Union[str, os.PathLike]) -> pd.DataFrame:
    print(f"\t\tLoading {csv_path} ...")
    return pd.read_csv(csv_path)


def emo_switch() -> dict:
    """Return events-AFNI emotion mappings."""
    return {
        "amusement": "Amu",
        "anger": "Ang",
        "anxiety": "Anx",
        "awe": "Awe",
        "calmness": "Cal",
        "craving": "Cra",
        "disgust": "Dis",
        "excitement": "Exc",
        "fear": "Fea",
        "horror": "Hor",
        "joy": "Joy",
        "neutral": "Neu",
        "romance": "Rom",
        "sadness": "Sad",
        "surprise": "Sur",
    }


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


class SupportFsl:
    """General helper methods for first- and second-level workflows."""

    def __init__(self, keoki_path: Union[str, os.PathLike]):
        """Initialize."""
        try:
            self._rsa_key = os.environ["RSA_LS2"]
        except KeyError as e:
            raise Exception(
                "Missing required environmental variable RSA_LS2"
            ) from e
        self._keoki_path = keoki_path

    @property
    def _ls2_ip(self):
        """Return labarserv2 ip addr."""
        return "ccn-labarserv2.vm.duke.edu"

    def _submit_rsync(self, src: str, dst: str) -> Tuple:
        """Execute rsync between DCC and labarserv2."""
        bash_cmd = f"""\
            rsync \
            -e "ssh -i {self._rsa_key}" \
            -rauv {src} {dst}
        """
        h_out, h_err = self._quick_sp(bash_cmd)
        return (h_out, h_err)

    def _quick_sp(self, bash_cmd: str) -> Tuple:
        """Spawn quick subprocess."""
        h_sp = subprocess.Popen(
            bash_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        h_out, h_err = h_sp.communicate()
        h_sp.wait()
        return (h_out, h_err)

    def _push_data(self):
        """Make remote destination and send data there."""
        keoki_dst = os.path.join(
            self._keoki_path, "derivatives", self._final_dir, self._subj
        )
        make_dst = f"""\
            ssh \
                -i {self._rsa_key} \
                {os.environ["USER"]}@{self._ls2_ip} \
                " command ; bash -c 'mkdir -p {keoki_dst}'"
            """
        _, _ = self._quick_sp(make_dst)

        # Send data
        dst = os.path.join(
            self._keoki_proj, "derivatives", self._final_dir, self._subj
        )
        _, _ = self._submit_rsync(self._subj_final, dst)


class SyncGroup(SupportFsl):
    """Coordinate setup, data download, output upload."""

    def __init__(self, work_deriv):
        """Initialize."""
        self._work_deriv = work_deriv
        self._keoki_path = (
            "/mnt/keoki/experiments2/EmoRep/"
            + "Exp2_Compute_Emotion/data_scanner_BIDS"
        )
        super().__init__(self._keoki_path)

    @property
    def _ls2_addr(self) -> str:
        """Return user@labarserv2."""
        return os.environ["USER"] + "@" + self._ls2_ip

    def setup_group(self) -> tuple:
        """Setup working directories for group models.

        Returns
        -------
        tuple
            - [0] = Path, work_deriv/model_afni
            - [1] = Path, work_deriv/model_afni_group

        """
        self._model_indiv = os.path.join(self._work_deriv, "model_afni")
        self._model_group = os.path.join(self._work_deriv, "model_afni_group")
        for _dir in [self._model_indiv, self._model_group]:
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        return (self._model_indiv, self._model_group)

    def get_model_afni(self, task: str, model_name: str):
        """Download model_afni data.

        Parameters
        ----------
        task: str
            {"task-movies", "task-scenarios"}
            BIDS task description
        model_name : str
            {"task", "block", "mixed"}
            AFNI deconvolution name

        """
        if task not in ["task-movies", "task-scenarios"]:
            raise ValueError(f"Unexpected task : {task}")
        if model_name not in ["task", "block", "mixed"]:
            raise ValueError(f"Unexpected model_name : {model_name}")
        if not hasattr(self, "_model_indiv"):
            self.setup_group()

        # Get all subject data
        source_afni = os.path.join(
            self._keoki_path, "derivatives", "model_afni", "sub-*"
        )
        bash_cmd = f"""rsync \
            -e "ssh -i {self._rsa_key}" \
            --prune-empty-dirs \
            --include "*/" \
            --include="*{task}_desc-decon_model-{model_name}*" \
            --exclude="*" \
            -rauv \
            {self._ls2_addr}:{source_afni} \
            {self._model_indiv}
        """
        h_out, h_err = self._quick_sp(bash_cmd)

        # Verify download
        chk_subj = glob.glob(
            f"{self._model_indiv}/sub-ER0009/ses-*/func/*{model_name}*"
        )
        if not chk_subj:
            print(h_out)
            print(h_err)
            raise FileNotFoundError("model_afni download failed")

    def send_group(self, test_dir: Union[str, os.PathLike]):
        """Send group output to Keoki."""
        keoki_dst = os.path.join(
            os.path.dirname(self._keoki_path), "analyses", "model_afni_group"
        )
        make_dst = f"""\
            ssh \
                -i {self._rsa_key} \
                {self._ls2_addr} \
                " command ; bash -c 'mkdir -p {keoki_dst}'"
        """
        _, _ = self._quick_sp(make_dst)
        _, _ = self._submit_rsync(test_dir, f"{self._ls2_addr}:{keoki_dst}")
