"""Methods for group-level analyses.

AfniExtractBetas : mine AFNI deconvolve output for betas estimates
EtacTest : conduct AFNI group-level test via 3dttest++
MvmTest : conduct AFNI group-level test via 3dMVM
ExtractTaskBetas : mine FSL first-level output for beta estimates
ImportanceMask : generate mask in template space from classifier output
ConjunctAnalysis : generate conjunction maps from ImportanceMask output

"""

import os
import sys
import re
import json
import glob
import time
import textwrap
from typing import Union, Tuple
import numpy as np
import pandas as pd
import subprocess
from multiprocessing import Pool
import nibabel as nib
from func_model.resources import helper
from func_model.resources import matrix
from func_model.resources import submit
from func_model.resources import sql_database


def _cmd_sub_int(
    sub_label: str,
    decon_path: Union[str, os.PathLike],
    out_txt: Union[str, os.PathLike] = None,
) -> list:
    """Return 3dinfo command to get sub-brick label int."""
    bash_cmd = ["3dinfo", "-label2index", f"'{sub_label}'", decon_path]
    if out_txt:
        bash_cmd.append(f"> {out_txt}")
    return bash_cmd


def get_subbrick_label(
    sub_label: str,
    model_name: str,
    decon_path: Union[str, os.PathLike],
    out_txt=None,
):
    """Title."""
    subj_work = os.path.dirname(decon_path)
    sing_head = helper.prepend_afni_sing(subj_work, subj_work)
    if not out_txt:
        out_txt = os.path.join(
            subj_work, f"subbrick_{model_name}_{sub_label.split('#')[0]}.txt"
        )
    sub_cmd = _cmd_sub_int(sub_label, decon_path, out_txt=out_txt)
    bash_cmd = " ".join(sing_head + sub_cmd)
    sub_job = subprocess.Popen(bash_cmd, shell=True, stdout=subprocess.PIPE)
    _, _ = sub_job.communicate()
    sub_job.wait()
    if not os.path.exists(out_txt):
        time.sleep(3)


class AfniExtractTaskBetas(matrix.NiftiArray):
    """Generate dataframe of voxel beta-coefficients.

    Split AFNI deconvolve files into desired sub-bricks, and then
    extract all voxel beta weights for sub-labels of interest.
    Convert extracted weights into a dataframe.

    Inherits general.matrix.NiftiArray.

    Methods
    -------
    get_label_names()
        Derive sub-brick identifiers
    get_label_int(sub_label)
        Determine specific sub-brick integer
    split_decon(emo_name=None)
        Split deconvolve file into each relevant sub-brick
    make_func_matrix(subj, sess, task, model_name, decon_path)
        Generate dataframe of voxel beta weights for subject, session

    Example
    -------
    etb_obj = group.ExtractTaskBetas(*args)
    etb_obj.mask_coord("/path/to/binary/mask/nii")
    df_path = etb_obj.make_func_matrix(*args)

    """

    def __init__(self, proj_dir, float_prec=4):
        """Initialize.

        Parameters
        ----------
        proj_dir : path
            Location of project directory
        float_prec : int, optional
            Desired float point precision of dataframes

        Raises
        ------
        TypeError
            Unexpected type for float_prec

        """
        print("Initializing ExtractTaskBetas")
        super().__init__(float_prec)
        self._proj_dir = proj_dir

    def get_label_names(self):
        """Get sub-brick levels from AFNI deconvolved file.

        Attributes
        ----------
        stim_label : list
            Full label ID of sub-brick starting with "mov" or "sce",
            e.g. movFea#0_Coef.

        Raises
        ------
        ValueError
            Trouble parsing output of 3dinfo -label to list

        Example
        -------
        etb_obj = group.ExtractTaskBetas(*args)
        etb_obj.decon_path = "/path/to/decon/file+tlrc"
        etb_obj.subj_out_dir = "/path/to/output/dir"
        etb_obj.get_label_names()

        """
        # Extract sub-brick label info
        out_label = os.path.join(self.subj_out_dir, "tmp_labels.txt")
        bash_list = [
            "3dinfo",
            "-label",
            self.decon_path,
            f"> {out_label}",
        ]
        bash_cmd = " ".join(bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_label, "Get labels")

        with open(out_label, "r") as lf:
            label_str = lf.read()
        label_list = [x for x in label_str.split("|")]
        if label_list[0] != "Full_Fstat":
            raise ValueError("Error in extracting decon labels.")
        os.remove(out_label)

        # Identify labels relevant to task
        self.stim_label = [
            x
            for x in label_list
            if self.task.split("-")[1][:3] in x and "Fstat" not in x
        ]
        self.stim_label.sort()

    def get_label_int(self, sub_label):
        """Return sub-brick ID of requested sub-brick label.

        Parameters
        ----------
        sub_label : str
            Sub-brick label, e.g. movFea#0_Coef

        Returns
        -------
        int

        Raises
        ------
        ValueError
            Trouble in parsing 3dinfo output

        Example
        -------
        etb_obj = group.ExtractTaskBetas(*args)
        etb_obj.decon_path = "/path/to/decon/file"
        etb_obj.subj_out_dir = "/path/to/output/dir"
        sub_int = etb_obj.get_label_int("movFea#0_Coef")

        """
        # Determine sub-brick integer value
        out_label = os.path.join(self.subj_out_dir, "tmp_label_int.txt")
        bash_cmd = " ".join(
            _cmd_sub_int(sub_label, self.decon_path, out_txt=out_label)
        )
        _ = submit.submit_subprocess(bash_cmd, out_label, "Label int")

        # Read label integer and check
        with open(out_label, "r") as lf:
            label_int = lf.read().strip()
        if len(label_int) > 2:
            raise ValueError(f"Unexpected int length for {sub_label}")
        os.remove(out_label)
        return label_int

    def split_decon(self, emo_name=None):
        """Split deconvolved files into files by sub-brick.

        Parameters
        ----------
        emo_name: str, optional
            Long name of emotion, for restricting beta extraction
            to single emotion

        Attributes
        ----------
        beta_dict : dict
            key = emotion, value = path to sub-brick file

        Raises
        ------
        ValueError
            Sub-brick identifier int has length > 2

        Example
        -------
        etb_obj = group.ExtractTaskBetas(**args)
        etb_obj.subj = "sub-100"
        etb_obj.sess = "ses-1"
        etb_obj.task = "task-movies"
        etb_obj.decon_path = "/path/to/decon/file+tlrc"
        etb_obj.subj_out_dir = "/path/to/output/dir"
        etb_obj.split_decon(emo_name="fear")

        """
        # Invert emo_switch to unpack sub-bricks
        _emo_switch = helper.emo_switch()
        emo_switch = {j: i for i, j in _emo_switch.items()}

        # Extract desired sub-bricks from deconvolve file by label name
        self.get_label_names()
        self.beta_dict = {}
        for sub_label in self.stim_label:
            # Identify emo string
            emo_long = emo_switch[sub_label[3:6]]
            if emo_name and emo_long != emo_name:
                continue

            # Determine sub-brick integer value
            label_int = self.get_label_int(sub_label)

            # Setup file name, avoid repeating beta extraction
            out_file = (
                f"tmp_{self.subj}_{self.sess}_{self.task}_"
                + f"desc-{emo_long}_beta.nii.gz"
            )
            out_path = os.path.join(self.subj_out_dir, out_file)
            if os.path.exists(out_path):
                self.beta_dict[emo_long] = out_path
                continue

            # Write sub-brick as new file
            print(f"\t\tExtracting sub-brick for {emo_long}")
            bash_list = [
                "3dTcat",
                f"-prefix {out_path}",
                f"{self.decon_path}[{label_int}]",
                "> /dev/null 2>&1",
            ]
            bash_cmd = " ".join(bash_list)
            _ = submit.submit_subprocess(bash_cmd, out_path, "Split decon")
            self.beta_dict[emo_long] = out_path

    def make_func_matrix(self, subj, sess, task, model_name, decon_path):
        """Generate a matrix of beta-coefficients from a deconvolve file.

        Extract task-related beta-coefficients maps, then vectorize the matrix
        and create a dataframe where each row has all voxel beta-coefficients
        for an event of interest.

        Dataframe is written to directory of <decon_path> and titled:
            <subj>_<sess>_<task>_desc-<model_name>_betas.tsv

        Parameters
        ----------
        subj : str
            BIDS subject identifier
        sess : str
            BIDS session identifier
        task : str
            [task-scenarios | task-movies]
            BIDS task identifier
        model_name : str
            Model identifier for deconvolution
        decon_path : path
            Location of deconvolved file

        Attributes
        ----------
        subj_out_dir : path
            Location for output files

        Returns
        -------
        path
            Location of output dataframe

        """
        # Check input, set attributes and output location
        if not helper.valid_task(task):
            raise ValueError(f"Unexpected value for task : {task}")

        print(f"\tGetting betas from {subj}, {sess}")
        self.subj = subj
        self.sess = sess
        self.task = task
        self.decon_path = decon_path
        self.subj_out_dir = os.path.dirname(decon_path)
        out_path = os.path.join(
            self.subj_out_dir,
            f"{subj}_{sess}_{task}_desc-{model_name}_betas.tsv",
        )
        if os.path.exists(out_path):
            return out_path

        # Generate/Get seprated decon behavior files
        self.split_decon()

        # Extract and vectorize voxel betas
        for emo, beta_path in self.beta_dict.items():
            print(f"\t\tExtracting betas for : {emo}")
            h_arr = self.nifti_to_arr(beta_path)
            img_arr = self.add_arr_id(subj, task, emo, h_arr)
            del h_arr

            # Create/update dataframe
            if "df_betas" not in locals() and "df_betas" not in globals():
                df_betas = self.arr_to_df(img_arr)
            else:
                df_tmp = self.arr_to_df(img_arr)
                df_betas = pd.concat(
                    [df_betas, df_tmp], axis=0, ignore_index=True
                )
                del df_tmp
            del img_arr

        # Clean if workflow uses mask_coord
        print("\tCleaning dataframe ...")
        if hasattr(self, "_rm_cols"):
            df_betas = df_betas.drop(self._rm_cols, axis=1)

        # Write and clean
        df_betas.to_csv(out_path, index=False, sep="\t")
        print(f"\t\tWrote : {out_path}")
        del df_betas
        tmp_list = glob.glob(f"{self.subj_out_dir}/tmp_*")
        for rm_file in tmp_list:
            os.remove(rm_file)
        return out_path


# %%
def comb_matrices(subj_list, model_name, proj_deriv, out_dir):
    """Combine participant beta dataframes into master.

    Find beta-coefficient dataframes for participants in subj_list
    and combine into a single dataframe.

    Output dataframe is written to:
        <out_dir>/afni_<model_name>_betas.tsv

    Parameters
    ----------
    subj_list : list
        Participants to include in final dataframe
    model_name : str
        Model identifier for deconvolution
    proj_deriv : path
        Location of project derivatives, will search for dataframes
        in <proj_deriv>/model_afni/sub-*.
    out_dir : path
        Output location of final dataframe

    Returns
    -------
    path
        Location of output dataframe

    Raises
    ------
    ValueError
        Missing participant dataframes

    """
    print("\tCombining participant beta tsv files ...")

    # Find desired dataframes
    df_list = sorted(
        glob.glob(
            f"{proj_deriv}/model_afni/**/*desc-{model_name}_betas.tsv",
            recursive=True,
        )
    )
    if not df_list:
        raise ValueError("No subject beta-coefficient dataframes found.")
    beta_list = [
        x for x in df_list if os.path.basename(x).split("_")[0] in subj_list
    ]

    # Combine dataframes and write out
    for beta_path in beta_list:
        print(f"\t\tAdding {beta_path} ...")
        if "df_betas_all" not in locals() and "df_betas_all" not in globals():
            df_betas_all = pd.read_csv(beta_path, sep="\t")
        else:
            df_tmp = pd.read_csv(beta_path, sep="\t")
            df_betas_all = pd.concat(
                [df_betas_all, df_tmp], axis=0, ignore_index=True
            )

    out_path = os.path.join(out_dir, f"afni_{model_name}_betas.tsv")
    df_betas_all.to_csv(out_path, index=False, sep="\t")
    print(f"\tWrote : {out_path}")
    return out_path


# %%
class EtacTest:
    """Build and execute ETAC tests.

    Identify relevant sub-bricks in AFNI's deconvolved files given user
    input, then construct and run ETAC tests. ETAC shell script and
    output files are written to <out_dir>.

    Parameters
    ----------
    out_dir : str, os.PathLike
        Output location for generated files
    mask_path : str, os.PathLike
        Location of group mask

    Methods
    --------
    write_exec(*args)
        Construct and run a T-test via ETAC (3dttest++)

    Example
    -------
    et_obj = group.EtacTest(*args)
    _ = et_obj.write_exec(*args)

    """

    def __init__(self, out_dir, mask_path):
        """Initialize."""
        self._out_dir = out_dir
        self._mask_path = mask_path

    def _etac_opts(
        self,
        out_path: Union[str, os.PathLike],
    ) -> list:
        """Return 3dttest++ call and ETAC options."""
        # Run 3dttest++ from singularity, account for CWD req
        work_dir = os.path.dirname(out_path)
        sing_cmd = helper.prepend_afni_sing(
            os.path.dirname(self._out_dir), work_dir
        )
        etac_head = [f"cd {work_dir};"] + sing_cmd + ["3dttest++"]

        # Build 3dttest++ command, account for paired vs student
        final_name = os.path.basename(out_path).split("+")[0]
        etac_body = [
            f"-mask {self._mask_path}",
            f"-prefix {final_name}",
            f"-prefix_clustsim {final_name}_clustsim",
            "-ETAC 12",
            "-ETAC_blur 0 2 4",
            "-ETAC_opt",
            "NN=2:sid=2:hpow=0:pthr=0.01,0.005,0.002,0.001:name=etac",
            "-Clustsim 12",
        ]
        if self._stat == "paired":
            return etac_head + ["-paired"] + etac_body
        else:
            return etac_head + etac_body

    def _setup_output(self, blk_coef: bool) -> Union[str, os.PathLike]:
        """Setup, return final filename and location."""
        # Make identifier string
        model = (
            f"model-{self._model_name}Block"
            if blk_coef
            else f"model-{self._model_name}"
        )
        id_str = f"stat-{self._stat}_{self._task}_{model}_emo-{self._emo_name}"

        # Setup output directory
        test_dir = os.path.join(self._out_dir, id_str)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        # Setup expected output
        return os.path.join(test_dir, f"{id_str}+tlrc.HEAD")

    def write_exec(
        self,
        task,
        model_name,
        stat,
        emo_name,
        decon_dict,
        sub_label,
        log_dir,
        blk_coef,
    ):
        """Write and execute a T-test using AFNI's ETAC method.

        Compare coefficient (sub_label) against null for stat=student
        or against the washout coefficient for stat=paired. Generates
        the ETAC command (3dttest++), writes it to a shell script for review,
        and then executes the command.

        Parameters
        ----------
        task : str
            {"task-movies", "task-scenarios"}
            BIDS task identifier
        model_name : str
            {"univ", "mixed"}
            Type of AFNI deconvolution
        stat : str
            {"student", "paired"}
            Type of T-test to conduct
        emo_name : str, key of afni.helper.emo_switch
            Lower case emotion name
        decon_dict : dict
            Group information in format:
            {"sub-ER0009": "/path/to/decon+tlrc.HEAD"}
        sub_label : str
            Sub-brick label, e.g. movFea#0_Coef
        log_dir : str, os.PathLike
            Location of logging files
        blk_coef : bool
            Use block (blk) coefficients rather than event, only
            available when model_name=mixed.

        Returns
        -------
        str, os.PathLike
            Location of final HEAD file

        """
        # Validate and setup
        self._task = task
        self._model_name = model_name
        self._stat = stat
        self._sub_label = sub_label
        self._decon_dict = decon_dict
        self._emo_name = emo_name
        self._validate_etac_input()
        out_path = self._setup_output(blk_coef)
        if os.path.exists(out_path):
            return

        # Make input list
        etac_set = [f"-setA {self._emo_name}"] + self._build_list()
        if stat == "paired":
            self._sub_label = "comWas#0_Coef"
            etac_set = etac_set + ["-setB washout"] + self._build_list()

        # Build ETAC command, write to script for review
        etac_list = self._etac_opts(out_path) + etac_set
        etac_cmd = " ".join(etac_list)
        with open(out_path.replace("+tlrc.HEAD", ".sh"), "w") as script:
            script.write(etac_cmd)

        # Execute ETAC command
        submit.submit_sbatch(
            etac_cmd,
            f"etac{self._emo_name}",
            log_dir,
            num_hours=75,
            num_cpus=12,
            mem_gig=32,
        )

        # Check for output
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Missing expected file : {out_path}")
        return out_path

    def _validate_etac_input(self):
        """Check that user input conforms."""
        # Check specified sub_label
        if self._sub_label.split("#")[1] != "0_Coef":
            raise ValueError("Improper format of sub_label")

        # Check model name
        if not helper.valid_univ_test(self._stat):
            raise ValueError("Improper model name specified")

    def _build_list(self) -> list:
        """Build ETAC set inputs of file paths and sub-brick labels."""
        # Iterate through dict and allow updating dict length
        set_list = []
        for subj in list(self._decon_dict):
            self._decon_path = self._decon_dict[subj]

            # Get sub-brick label, remove subj from decon_dict
            # if missing sub-brick. This allows for a balanced
            # emo_coef vs washout_coef.
            label_int = self._get_label_int()
            if not label_int:
                self._decon_dict.pop(subj)
                continue

            # Build input line
            set_list.append(subj)
            set_list.append(f"{self._decon_path}'[{label_int}]'")
        return set_list

    def _get_label_int(self) -> Union[str, None]:
        """Return sub-brick num given label."""
        subbrick_txt = os.path.join(
            os.path.dirname(self._decon_path),
            f"subbrick_{self._model_name}_{self._sub_label.split('#')[0]}.txt",
        )

        # Extract sub-brick label num if needed
        if (
            not os.path.exists(subbrick_txt)
            or os.stat(subbrick_txt).st_size == 0
        ):
            get_subbrick_label(
                self._sub_label,
                self._model_name,
                self._decon_path,
                out_txt=subbrick_txt,
            )
        if not os.path.exists(subbrick_txt):
            return None

        # Get last line from subbrick_txt
        with open(subbrick_txt) as f:
            for line in f:
                pass
            try:
                last_line = line.strip()
            except UnboundLocalError:
                print(f"\tFailed on {self._sub_label} for {self._decon_path}")
                return None

        # Validate line has content
        if len(last_line) == 0:
            return None
        return last_line


def _blur_decon(model_indiv, mask_path, blur, decon_list):
    """Title."""
    try:
        arr_id = os.environ["SLURM_ARRAY_TASK_ID"]
    except KeyError:
        raise EnvironmentError(
            "group._blur_decon intended for execution by SLURM array"
        )

    decon_path = decon_list[int(arr_id)]
    out_path = decon_path.replace("+tlrc", f"_blur-{blur}+tlrc")
    if os.path.exists(out_path):
        return

    sing_cmd = helper.prepend_afni_sing(
        os.path.dirname(mask_path), model_indiv
    )
    afni_blur = [
        "3dmerge",
        f"-prefix {out_path}",
        f"-1blur_fwhm {blur}",
        "-doall",
        decon_path,
    ]
    afni_cmd = " ".join(sing_cmd + afni_blur)
    _ = submit.submit_subprocess(afni_cmd, out_path, "blur")


def _calc_acf(model_indiv, mask_path, errts_list, out_txt):
    """Title."""
    try:
        arr_id = os.environ["SLURM_ARRAY_TASK_ID"]
    except KeyError:
        raise EnvironmentError(
            "group._calc_acf intended for execution by SLURM array"
        )

    #
    errts_path = errts_list[int(arr_id)]
    out_path = os.path.join(os.path.dirname(errts_path), out_txt)
    if os.path.exists(out_path):
        return

    #
    sing_cmd = helper.prepend_afni_sing(
        os.path.dirname(mask_path), os.path.dirname(errts_path)
    )
    afni_blur = [
        "3dFWHMx",
        f"-mask {mask_path}",
        f"-input {errts_path}",
        "-acf",
        f"> {out_path}",
    ]
    afni_cmd = " ".join(sing_cmd + afni_blur)
    # _ = submit.submit_subprocess(afni_cmd, out_path, "3dfwhmx")
    job_sp = subprocess.Popen(
        afni_cmd,
        shell=True,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
    )
    # job_out, job_err = job_sp.communicate()
    job_sp.wait()


class MvmTest:
    """Build and execute 3dMVM tests.

    Construct a two-factor repeated-measures ANOVA for sanity checking.
    3dMVM scripts and output are written to <out_dir>.

    Parameters
    ----------

    Methods
    --------
    clustsim()
        Conduct Monte Carlo simulations on group-level mask
    write_exec(*args)
        Generate the 3dMVM command and then execute

    Example
    -------
    mvm_obj = group.MvmTest(*args)
    mvm_obj.clustsim()
    _ = mvm_obj.write_exec(*args)

    """

    def __init__(
        self,
        model_indiv,
        model_group,
        mask_path,
        log_dir,
        use_blur=True,
        blur=4,
    ):
        """Initialize."""
        self._model_indiv = model_indiv
        self._model_group = model_group
        self._mask_path = mask_path
        self._log_dir = log_dir
        self._use_blur = use_blur
        self._blur = blur

    def _build_row(self, row_dict: dict):
        """Write single row of datatable."""
        # Identify sub-brick integer from label
        for _, info_dict in row_dict.items():
            self._get_subbrick.decon_path = info_dict["decon_path"]
            self._get_subbrick.subj_out_dir = self._model_group
            label_int = self._get_subbrick.get_label_int(
                info_dict["sub_label"]
            )

            # Write line
            self._table_list.append(info_dict["subj"])
            self._table_list.append(info_dict["sesslabel"])
            self._table_list.append(info_dict["stimlabel"])
            self._table_list.append(
                f"{info_dict['decon_path']}'[{label_int}]'"
            )

    def _build_table(self):
        """Build the datatable of 3dMVM."""
        # Initialize ExtractTaskBetas, subset group_dict for
        # individual subjects, sessions
        self._get_subbrick = ExtractTaskBetas(self._model_indiv)
        self._table_list = []
        for subj in self._group_dict:
            for task, info_dict in self._group_dict[subj].items():
                # Organize session data to write mulitple lines
                stim = task.split("-")[1]
                row_dict = {
                    "emo": {
                        "subj": subj,
                        "sesslabel": stim,
                        "stimlabel": "emo",
                        "decon_path": info_dict["decon_path"],
                        "sub_label": info_dict["emo_label"],
                    },
                    "base": {
                        "subj": subj,
                        "sesslabel": stim,
                        "stimlabel": "wash",
                        "decon_path": info_dict["decon_path"],
                        "sub_label": info_dict["wash_label"],
                    },
                }
                self._build_row(row_dict)

    def write_exec(self, group_dict, model_name, emo_short):
        """Write and execute 3dMVM command.

        Build a repeated-measures ANOVA, using two within-subject
        factors (sesslabel, stimlabel). Output F-stats and posthoc
        stimlabel emo-wash T-stat.

        Parameters
        ----------
        group_dict : dict
            All participant data, in format:
                - first-level keys = BIDS subject identifier
                - second-level keys = BIDS task identifier
                - third-level dictionary:
                    - sess: BIDS session identifier
                    - decon_path: path to decon file
                    - emo_label:  emotion subbrick label
                    - wash_label: washout subbrick label
            Example:
                {"sub-ER0009": {
                    "task-movies": {
                        "sess": "ses-day2",
                        "decon_path": "/path/to/ses-day2/decon",
                        "emo_label": "movFea#0_Coef",
                        "wash_label": "comWas#0_Coef",
                        },
                    "scenarios": {},
                    },
                }
        model_name : str
            Model identifier
        emo_short : str
            Shortened (AFNI) emotion name

        """
        # Validate
        if not helper.valid_mvm_test(model_name):
            raise ValueError(f"Unexpected model_name : {model_name}")
        first_keys = list(group_dict.keys())
        if not any("sub-ER" in x for x in first_keys):
            raise KeyError("First-level key not matching format : sub-ER*")
        second_keys = list(group_dict[first_keys[0]].keys())
        for task in second_keys:
            if not helper.valid_task(task):
                raise KeyError(f"Unexpected second-level key : {task}")
        third_keys = list(group_dict[first_keys[0]][second_keys[0]].keys())
        valid_keys = ["sess", "decon_path", "emo_label", "wash_label"]
        for key in third_keys:
            if key not in valid_keys:
                raise KeyError(f"Unexpected third-level key : {key}")

        # Setup, check for existing output
        self._group_dict = group_dict
        final_name, out_path = self._setup_output(model_name, emo_short)
        if os.path.exists(out_path):
            return self._out_dir

        print("\tBuilding 3dMVM command")
        mvm_head = [
            "3dMVM",
            f"-prefix {self._model_group}/{final_name}",
            "-jobs 12",
            "-bsVars 1",
            "-wsVars 'sesslabel*stimlabel'",
            f"-mask {self._mask_path}",
            "-num_glt 2",
            "-gltLabel 1 movie_vs_scen",
            "-gltCode 1 'sesslabel : 1*movies -1*scenarios'",
            "-gltLabel 2 emo_vs_wash",
            "-gltCode 2 'stimlabel : 1*emo -1*wash'",
            "-dataTable",
            "Subj sesslabel stimlabel InputFile",
        ]
        self._build_table()
        mvm_cmd = " ".join(mvm_head + self._table_list)
        self._write_script(final_name, mvm_cmd)

        print("\tExecuting 3dMVM command")
        submit.submit_subprocess(mvm_cmd, out_path, f"mvm{emo_short}")

    def _submit_subproc(self, job_cmd: str):
        """Title."""
        job_sp = subprocess.Popen(
            job_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        job_out, job_err = job_sp.communicate()
        job_sp.wait()
        return (job_out, job_err)

    def blur_decon(self, model_name: str):
        """Title.

        Attributes
        ----------
        blur_dict : dict

        """
        # Manually build decon_list to avoid resubmitting entire array
        # for every single file update.
        search_path = os.path.join(self._model_indiv, "sub-*", "ses-*", "func")
        decon_list = []
        func_dirs = sorted(glob.glob(search_path))
        for func_path in func_dirs:
            for file_type in ["stats", "errts"]:

                #
                chk_stat = (
                    f"{func_path}/decon_{model_name}_"
                    + f"{file_type}_REML+tlrc.HEAD"
                )
                chk_blur = chk_stat.replace(
                    "+tlrc", f"_blur-{self._blur}+tlrc"
                )
                if os.path.exists(chk_stat) and not os.path.exists(chk_blur):
                    decon_list.append(chk_stat)

        if not decon_list:
            return

        #
        sbatch_cmd = f"""\
            #!/bin/env {sys.executable}

            #SBATCH --output={self._log_dir}/blur_decon_%a.txt
            #SBATCH --array=0-{len(decon_list) - 1}%20
            #SBATCH --time=06:00:00
            #SBATCH --mem=24G
            #SBATCH --wait

            from func_model.resources import group
            group._blur_decon(
                "{self._model_indiv}",
                "{self._mask_path}",
                {self._blur},
                {decon_list},
            )

        """
        sbatch_cmd = textwrap.dedent(sbatch_cmd)

        #
        py_script = f"{self._log_dir}/run_blur_decon.py"
        with open(py_script, "w") as ps:
            ps.write(sbatch_cmd)
        self._submit_subproc(f"sbatch {py_script}")

    def noise_acf(self, model_name):
        """Title."""
        #
        search_path = os.path.join(self._model_indiv, "sub-*", "ses-*", "func")
        search_str = (
            f"decon_{model_name}_errts_REML_blur-{self._blur}+tlrc.HEAD"
            if self._use_blur
            else f"decon_{model_name}_errts_REML+tlrc.HEAD"
        )
        errts_all = sorted(glob.glob(f"{search_path}/{search_str}"))
        if not errts_all:
            raise FileNotFoundError(
                f"Glob failed at finding : {search_path}/{search_str}"
            )

        # Build errts_list to avoid submitting large array for few
        # needed files.
        errts_list = []
        self._acf_name = (
            f"ACF_raw_blur{self._blur}.txt"
            if self._use_blur
            else "ACF_raw.txt"
        )
        for errts_path in errts_all:
            chk_out = os.path.join(os.path.dirname(errts_path), self._acf_name)
            if not os.path.exists(chk_out):
                errts_list.append(errts_path.split(".")[0])

        if not errts_list:
            return

        #
        sbatch_cmd = f"""\
            #!/bin/env {sys.executable}

            #SBATCH --output={self._log_dir}/acf_errts_%a.txt
            #SBATCH --array=0-{len(errts_list) - 1}%20
            #SBATCH --time=06:00:00
            #SBATCH --mem=24G
            #SBATCH --wait

            from func_model.resources import group
            group._calc_acf(
                "{self._model_indiv}",
                "{self._mask_path}",
                {errts_list},
                "{self._acf_name}",
            )

        """
        sbatch_cmd = textwrap.dedent(sbatch_cmd)

        #
        py_script = f"{self._log_dir}/run_calc_acf.py"
        with open(py_script, "w") as ps:
            ps.write(sbatch_cmd)
        self._submit_subproc(f"sbatch {py_script}")

    def clustsim(self):
        """Conduct mask-based Monte Carlo simulations."""
        sim_out = os.path.join(self._model_group, "montecarlo_simulations.txt")
        if os.path.exists(sim_out):
            return sim_out

        # Calculate average ACF
        self._get_acf()
        mean_acf = list(self._df_acf.mean())

        # Build and submit cluststim
        afni_prep = helper.prepend_afni_sing(
            self._model_group, self._model_indiv
        )
        bash_list = [
            "3dClustSim",
            f"-mask {self._mask_path}",
            "-LOTS",
            "-iter 10000",
            "-pthr 0.001",
            "-athr 0.05",
            f"-acf {mean_acf[0]} {mean_acf[1]} {mean_acf[2]}",
            f"> {sim_out}",
        ]
        submit.submit_subprocess(
            " ".join(afni_prep + bash_list), sim_out, "MCsim"
        )

    def _get_acf(self):
        """Title."""
        search_path = os.path.join(self._model_indiv, "sub-*", "ses-*", "func")
        acf_list = sorted(glob.glob(f"{search_path}/{self._acf_name}"))
        if not acf_list:
            raise FileNotFoundError(
                f"Glob failed at finding : {search_path}/{self._acf_name}"
            )

        #
        with Pool(processes=8) as pool:
            items = [(x) for x in acf_list]
            acf_values = pool.map(self._mine_acf, items)
        self._df_acf = pd.DataFrame(
            acf_values, columns=["x", "y", "z", "comb"]
        )

        # Write out for
        out_acf = os.path.join(self._model_group, "acf_all.csv")
        self._df_acf.to_csv(out_acf)

    def _mine_acf(self, acf_path) -> list:
        """Title."""
        with open(acf_path) as f:
            for line in f:
                pass
            last_line = line

        acf_values = [float(x) for x in last_line.strip().split()]
        if len(acf_values) == 4:
            return acf_values


# %%


class _GetCopes:
    """Mine design.con to return {emo: nii} dict.

    Methods
    -------
    find_copes()
        Find copes and return {emo: nii} mapping

    """

    def __init__(self, con_name: str):
        """Initialize."""
        self._con_name = con_name

    def find_copes(self, design_path: Union[str, os.PathLike]) -> dict:
        """Match emotion name to cope file, return {emo: nii}."""
        self._design_path = design_path

        # Mine, organize design.con info
        self._read_contrast()
        self._drop_contrast()
        self._clean_contrast()

        # Orient from design.con to stats dir
        feat_dir = os.path.dirname(self._design_path)
        stats_dir = os.path.join(feat_dir, "stats")

        # Match emotion to nii path
        cope_dict = {}
        for emo, num in self._con_dict.items():
            nii_path = os.path.join(stats_dir, f"cope{num}.nii.gz")
            if not os.path.exists(nii_path):
                raise FileNotFoundError(
                    f"Missing expected cope file : {nii_path}"
                )
            cope_dict[emo] = nii_path
        return cope_dict

    def _read_contrast(self):
        """Match contrast name to number."""
        # Extract design.con lines starting with /ContrastName
        con_lines = []
        with open(self._design_path) as dp:
            for ln in dp:
                if ln.startswith("/ContrastName"):
                    con_lines.append(ln[1:])
        if len(con_lines) == 0:
            raise ValueError(
                "Could not find ContrastName in " + f"{self._design_path}"
            )

        # Organize contrast lines
        self._con_dict = {}
        for line in con_lines:
            con, name = line.split()
            self._con_dict[name] = con

    def _drop_contrast(self):
        """Drop contrasts of no interest."""
        for key in list(self._con_dict.keys()):
            if not key.startswith(self._con_name):
                del self._con_dict[key]

    def _clean_contrast(self):
        """Clean emotion: contrast pairs."""
        out_dict = {}
        clean_num = len(self._con_name)
        for key, value in self._con_dict.items():
            new_key = key[clean_num:].split("GT")[0].lower()
            out_dict[new_key] = value[-1]
        self._con_dict = out_dict


class ExtractTaskBetas(matrix.NiftiArray):
    """Generate dataframe of voxel beta-coefficients.

    Align FSL cope.nii files with their corresponding contrast
    names and then extract all voxel beta weights for contrasts
    of interest. Converts extracted beta weights into a dataframe.

    Extracted beta weights are sent to a tbl_betas_* in db_emorep,
    and are also written to Keoki. Inherits
    func_model.resources.general.matrix.NiftiArray.

    Methods
    -------
    make_beta_matrix(*args)
        Identify and align cope.nii files, mine for betas
        and generate dataframe

    Example
    -------
    etb_obj = group.ExtractTaskBetas()
    etb_obj.mask_coord("/path/to/binary/mask.nii")
    etb_obj.make_beta_matrix(*args)

    """

    def __init__(self):
        """Initialize."""
        print("Initializing ExtractTaskBetas")
        super().__init__()

    def make_beta_matrix(
        self,
        subj,
        sess,
        task,
        model_name,
        model_level,
        con_name,
        design_list,
        subj_out,
        overwrite,
        mot_thresh=0.2,
        max_value=9999,
    ):
        """Generate a matrix of beta-coefficients from FSL GLM cope files.

        Extract task-related beta-coefficients maps, then vectorize the matrix
        and create a dataframe where each row has all voxel beta-coefficients
        for an event of interest.

        Parameters
        ----------
        subj : str
            BIDS subject identifier
        sess : str
            BIDS session identifier
        task : str
            {"task-movies", "task-scenarios"}
            BIDS task identifier
        model_name : str
            {"lss", "sep"}
            FSL model identifier
        model_level : str
            {"first"}
            FSL model level
        con_name : str
            {"stim", "tog"}
            Desired contrast from which coefficients will be extracted
        design_list : list
            Paths to participant design.con files
        subj_out : path
            Output location for betas dataframe
        overwrite : bool
            Whether to overwrite existing data
        mot_thresh : float, optional
            Runs with a proportion of volumes >= mot_thresh will
            not be included in output dataframe
        max_value : int, optional
            Maximum expected beta value, anything > max_value or
            < -max_value will be censored.

        Notes
        -----
        - model_name="sep" requires con_name="stim"
        - model_name="lss" requires con_name="tog"

        """
        # Validate args and setup
        if not helper.valid_task(task):
            raise ValueError(f"Unexpected value for task : {task}")
        if model_name not in ["sep", "lss"]:
            raise ValueError(
                f"Unsupported value for model_name : {model_name}"
            )
        if model_level not in ["first"]:
            raise ValueError(
                f"Unsupported value for model_level : {model_level}"
            )
        if con_name not in ["stim", "tog"]:
            raise ValueError(f"Unsupported value for con_name : {con_name}")
        if model_name == "sep" and con_name != "stim":
            raise ValueError(
                "Unexpected model contrast pair : "
                + f"{model_name}, {con_name}"
            )
        if model_name == "lss" and con_name != "tog":
            raise ValueError(
                "Unexpected model contrast pair : "
                + f"{model_name}, {con_name}"
            )

        self._subj = subj
        self._sess = sess
        self._task = task
        self._model_name = f"name-{model_name}"
        self._model_level = f"level-{model_level}"
        self._con_name = con_name
        self._subj_out = subj_out
        self._overwrite = overwrite
        self._mot_thresh = mot_thresh
        self._max_value = max_value

        # Manage sep vs lss float precision
        if model_name == "lss":
            self._float_prec = 3

        # Check if records already exist in db_emorep
        print(f"Working on {subj}, {task}, {model_name}, {con_name}")
        data_exist = self._check_exist()
        if not self._overwrite and data_exist:
            print(
                f"\tData already exist for {subj}, {task}, "
                + f"{model_name}, {con_name}; Continuing ..."
            )
            return

        # Orient to design.con files and mine niis in parallel
        self._get_copes = _GetCopes(con_name)
        with Pool(processes=8) as pool:
            items = [(x) for x in design_list]
            self._data_obj = pool.map(self._mine_copes, items)
        if not isinstance(self._data_obj[0], tuple):
            return

        # Write csv and update db_emorep
        # TODO deprecate _write_csv()
        if model_name != "lss":
            self._write_csv()

        if model_name == "lss":
            self._update_fsl_betas_lss()
        else:
            self._update_fsl_betas_reg()
        del self._data_obj

    def _check_exist(self) -> bool:
        """Return bool of whether record exists in db_emorep."""
        # Using separate connection here from _update_db_emorep
        # to avoid multiproc pickle issue.
        db_con = sql_database.DbConnect()
        update_betas = sql_database.DbUpdateBetas(db_con)
        data_exist = update_betas.check_db(
            self._subj,
            self._task,
            self._model_name.split("-")[-1],
            self._con_name,
        )
        db_con.close_con()
        return data_exist

    def _mine_copes(
        self,
        design_path: Union[str, os.PathLike],
    ) -> Tuple:
        """Vectorize cope betas, return tuple of pd.DataFrame, run number."""
        # Determine run number for file name
        _run_dir = os.path.basename(os.path.dirname(design_path))
        run_num = _run_dir.split("_")[0].split("-")[1]
        run = f"run-{run_num}"
        if len(run_num) != 2:
            raise ValueError("Error parsing path for run number")

        # Compare proportion of outliers to criterion, skip run
        # if the threshold is exceeded.
        prop_path = os.path.join(
            self._subj_out,
            "confounds_proportions",
            f"{self._subj}_{self._sess}_{self._task}_{run}_"
            + "desc-confounds_proportion.json",
        )
        if not os.path.exists(prop_path):
            raise FileNotFoundError(f"Expected to find : {prop_path}")
        with open(prop_path) as jf:
            prop_dict = json.load(jf)
        if prop_dict["CensorProp"] >= self._mot_thresh:
            return

        # Find and match copes to emotions, get voxel betas
        print(f"\tGetting betas from {self._subj}, {self._task}, {run}")
        cope_dict = self._get_copes.find_copes(design_path)
        for emo, cope_path in cope_dict.items():
            img_arr = self.nifti_to_arr(cope_path)

            # Create/update dataframe
            if "df_out" not in locals():
                df_out = self.arr_to_df(img_arr, f"emo_{emo}")
            else:
                df_tmp = self.arr_to_df(img_arr, f"emo_{emo}")
                df_out[f"emo_{emo}"] = df_tmp[f"emo_{emo}"]
                del df_tmp

        # Censor extreme outliers and return df and run num
        # (to ensure proper order from multiproc).
        df_out = df_out.reset_index()
        emo_list = [x for x in df_out.columns if "emo" in x]
        df_out[
            (df_out[emo_list] > self._max_value)
            | (df_out[emo_list] < -self._max_value)
        ] = np.nan
        return (df_out, int(run_num))

    def _write_csv(self):
        """Write pd.DataFrame to disk."""
        # Unpack data_obj
        for idx, _ in enumerate(self._data_obj):
            if not isinstance(self._data_obj[idx], tuple):
                continue
            df = self._data_obj[idx][0]
            run = self._data_obj[idx][1]

            # Setup output path, write
            out_path = os.path.join(
                self._subj_out,
                f"{self._subj}_{self._sess}_{self._task}_run-0{run}_"
                + f"{self._model_level}_{self._model_name}_"
                + f"con-{self._con_name}_betas.csv",
            )
            print(f"\tWriting : {out_path}")
            df.to_csv(out_path, index=False)

    def _update_fsl_betas_reg(self):
        """Update db_emorep beta table for sep, tog models."""

        # Start left dfs of proper length, for each exposure
        df_a = self._data_obj[0][0][["voxel_name"]].copy()
        df_b = self._data_obj[0][0][["voxel_name"]].copy()
        df_a["num_block"] = 1
        df_b["num_block"] = 2

        # Unpack data_obj
        for idx, _ in enumerate(self._data_obj):
            if not isinstance(self._data_obj[idx], tuple):
                continue

            # Merge data with appropriate left df
            df = self._data_obj[idx][0]
            run_num = self._data_obj[idx][1]
            if run_num < 5:
                df_a = df_a.merge(df, how="left", on="voxel_name")
            else:
                df_b = df_b.merge(df, how="left", on="voxel_name")

        # Each proc gets own connection
        db_con = sql_database.DbConnect()
        update_betas = sql_database.DbUpdateBetas(db_con)

        # Update db_emorep beta table
        if df_a.shape[1] > 2:
            update_betas.update_db(
                df_a,
                self._subj,
                self._task,
                self._model_name.split("-")[-1],
                self._con_name,
                self._overwrite,
            )
        if df_b.shape[1] > 2:
            update_betas.update_db(
                df_b,
                self._subj,
                self._task,
                self._model_name.split("-")[-1],
                self._con_name,
                self._overwrite,
            )
        db_con.close_con()

    def _update_fsl_betas_lss(self):
        """Update db_emorep beta table for lss models."""

        def _mk_df() -> pd.DataFrame:
            """Return pd.DataFrame with voxel_name, num_event cols."""
            # Start with wide format
            df_par = self._data_obj[0][0][["voxel_name"]].copy()
            df_par["ev1"] = 1
            df_par["ev2"] = 2
            df_par["ev3"] = 3
            df_par["ev4"] = 4
            df_par["ev5"] = 5

            # Conver to long, clean redundant columns
            df = pd.wide_to_long(
                df_par, stubnames="ev", i="voxel_name", j="event"
            )
            df = df.reset_index()
            df = df.rename(columns={"event": "num_event"})
            df = df.drop(["ev"], axis=1)
            return df

        def _add_df(
            emo_name: str, df_l: pd.DataFrame, df_r: pd.DataFrame
        ) -> pd.DataFrame:
            """Update df_l with df_r data in col emo_name"""
            if emo_name not in df_l.columns:
                df_l = df_l.merge(
                    df_r,
                    how="left",
                    on=["voxel_name", "num_event"],
                )
            else:
                df_l = df_l.merge(
                    df_r,
                    how="left",
                    on=["voxel_name", "num_event"],
                )
                df_l[emo_name] = df_l[f"{emo_name}_y"].fillna(
                    df_l[f"{emo_name}_x"]
                )
                df_l = df_l.drop([f"{emo_name}_x", f"{emo_name}_y"], axis=1)
            return df_l

        # Start dfs for first, second emo block
        df_a = _mk_df()
        df_a["num_block"] = 1
        df_b = _mk_df()
        df_b["num_block"] = 2

        # Unpack data_obj
        for idx, _ in enumerate(self._data_obj):
            if not isinstance(self._data_obj[idx], tuple):
                continue
            df = self._data_obj[idx][0]
            run_num = self._data_obj[idx][1]

            # Identify emotion name and event number
            col_name = df.columns[1]
            emo_name = col_name.split("_ev")[0]
            ev_num = int(col_name[-1])

            # Prep df for merge, update parent dfs
            df = df.rename(columns={col_name: emo_name})
            df["num_event"] = ev_num
            if run_num < 5:
                df_a = _add_df(emo_name, df_a, df)
            else:
                df_b = _add_df(emo_name, df_b, df)

        # Convert missing data for SQL
        df_a = df_a.replace({np.nan: None})
        df_b = df_b.replace({np.nan: None})

        # Each proc gets own connection
        db_con = sql_database.DbConnect()
        update_betas = sql_database.DbUpdateBetas(db_con)

        # Update db_emorep beta table
        if df_a.shape[1] > 3:
            update_betas.update_db(
                df_a,
                self._subj,
                self._task,
                self._model_name.split("-")[-1],
                self._con_name,
                self._overwrite,
            )
        if df_b.shape[1] > 3:
            update_betas.update_db(
                df_b,
                self._subj,
                self._task,
                self._model_name.split("-")[-1],
                self._con_name,
                self._overwrite,
            )
        db_con.close_con()


# %%
class _MapMethods:
    """Voxel importance map methods.

    Methods
    -------
    c3d_add()
        Add voxels maps together
    cluster()
        Identify clusters of voxel value, size, and NN

    """

    def c3d_add(
        self,
        add_list: list,
        out_path: Union[str, os.PathLike],
    ):
        """Add 3D NIfTIs together."""
        bash_cmd = f"""\
            c3d \
                {" ".join(add_list)} \
                -accum -add -endaccum \
                -o {out_path}
        """
        _ = submit.submit_subprocess(bash_cmd, out_path, "c3d-add")
        self._tpl_head(out_path)

    def _tpl_head(self, out_path):
        """Fix header of conjunction files."""
        _ = submit.submit_subprocess(
            f"3drefit -space MNI {out_path}", out_path, "afni-refit"
        )

    def cluster(
        self,
        in_path: Union[str, os.PathLike],
        nn: int = 1,
        size: int = 10,
        vox_value: int = 2,
    ):
        """Identify clusters of NN, size, and voxel value."""
        out_dir = os.path.dirname(in_path)
        out_name = "Clust_" + os.path.basename(in_path)
        out_path = os.path.join(out_dir, out_name)
        bash_cmd = f"""\
            3dClusterize \
                -nosum -1Dformat \
                -inset {in_path} \
                -idat 0 -ithr 0 \
                -NN {nn} \
                -clust_nvox {size} \
                -bisided -{vox_value} {vox_value} \
                -pref_map {out_path} \
                > {out_path.replace(".nii.gz", ".txt")}
        """
        _ = submit.submit_subprocess(
            bash_cmd, out_path, "afni-clust", force_cont=True
        )


class ImportanceMask(matrix.NiftiArray, _MapMethods):
    """Convert a dataframe of classifier values into a NIfTI mask.

    Reference a template to derive header information and start a
    matrix of the same size. Populate said matrix was row values
    from a supplied dataframe.

    Inherits general.matrix.NiftiArray, _MapMethods

    Parameters
    ----------
    tpl_path : path
        Location and name of template

    Methods
    -------
    emo_names()
        Return list of emotion names in db_emorep.ref_emo
    make_mask(df, mask_path)
        Deprecated.
        Turn dataframe of values into NIfTI mask
    sql_masks()
        Pull data from db_emorep.tbl_plsda_* to
        generate masks

    Example
    -------
    im_mask = group.ImportanceMask(Path)
    emo_list = im_mask.emo_names()
    for emo in emo_list:
        mask_path = im_mask.sql_masks(*args)

    """

    def __init__(self, tpl_path):
        """Initialize."""
        print("Initializing ImportanceMask")
        super().__init__()
        if not os.path.exists(tpl_path):
            raise FileNotFoundError(f"Missing file : {tpl_path}")
        self._tpl_path = tpl_path
        self._mine_template()

    def _mine_template(self):
        """Mine a NIfTI template for information.

        Capture the NIfTI header and generate an empty
        np.ndarray of the same size as the template.

        Attributes
        ----------
        _img_header : obj, nibabel.nifti1.Nifti1Header
            Header data of template
        _empty_matrix : np.ndarray
            Matrix containing zeros that is the same size as
            the template.

        """
        print(f"\tMining domain info from : {self._tpl_path}")
        img = self.nifti_to_img(self._tpl_path)
        img_data = img.get_fdata()
        self._img_header = img.header
        self._empty_matrix = np.zeros(img_data.shape)

    def make_mask(self, df, mask_path, task_name):
        """Convert row values into matrix and save as NIfTI mask.

        Deprecated.

        Using the dataframe column names, fill an empty matrix
        with row values and then save the file as a NIfTI mask.

        Parameters
        ----------
        df : pd.DataFrame
            A header and single row containing classifier importance.
            Column names should be formatted as coordinate, e.g.
            "(45, 31, 90)".
        mask_path : str, os.PathLike
            Location and name of output NIfTI file
        task_name : str
            Name of stimulus type

        Returns
        -------
        nd.array
            Matrix of template size filled with classifier
            importance values.

        Raises
        ------
        AttributeError
            Missing required attributes
        KeyError
        ValueError
            Improper formatting of dataframe

        """
        return

        # Check for required attrs
        if not hasattr(self, "empty_matrix") and not hasattr(
            self, "img_header"
        ):
            raise AttributeError(
                "Attributes empty_matrix, img_header "
                + "required. Try ImportanceMask.mine_template."
            )

        # Validate dataframe
        if df.shape[0] != 1:
            raise ValueError("Dataframe must have only one row")
        chk_col = df.columns[0]
        try:
            int(re.sub("[^0-9]", "", chk_col))
        except ValueError:
            raise KeyError("Improperly formatted df column name.")
        if len(re.sub("[^0-9]", " ", chk_col).split()) != 3:
            raise KeyError("Improperly formatted df column name.")

        # Convert column names into a list of coordinate values
        print(f"\tBuilding importance map : {mask_path}")
        arr_fill = self.empty_matrix.copy()
        col_emo = [re.sub("[^0-9]", " ", x).split() for x in df.columns]

        # Add each column's value to the appropriate coordinate
        # in the empty matrix.
        for col_idx in col_emo:
            x = int(col_idx[0])
            y = int(col_idx[1])
            z = int(col_idx[2])
            arr_fill[x][y][z] = df.loc[0, f"({x}, {y}, {z})"]

        # Write matrix as a nii, embed template header
        emo_img = nib.Nifti1Image(
            arr_fill, affine=None, header=self.img_header
        )
        nib.save(emo_img, mask_path)
        clust_size = 5 if task_name == "scenarios" else 10
        self.cluster(mask_path, size=clust_size, vox_value=1)
        return arr_fill

    def emo_names(self) -> list:
        """Return list of emotions in db_emorep.ref_emo."""
        db_con = sql_database.DbConnect()
        emo_list = list(
            db_con.fetch_df(
                "select emo_name from ref_emo", ["emo_name"]
            ).emo_name
        )
        db_con.close_con()
        return emo_list

    def sql_masks(
        self,
        task_name,
        model_name,
        con_name,
        emo_name,
        binary_importance,
        out_dir,
        cluster=False,
    ):
        """Generate mask from db_emorep.tbl_plsda_* data.

        Parameters
        ----------
        task_name : str
            {"movies", "scenarios", "both"}
            Classifier task
        model_name : str
            {"sep", "tog", "rest", "lss"}
            FSL model name
        con_name : str
            {"stim", "tog", "replay"}
            FSL contrast name
        emo_name : str
            Emotion name e.g. "awe" or "amusement"
        binary_importance : str
            {"binary", "importance"}
            Used to select tbl_plsda_*
        out_dir : str, os.PathLike
            Output directory path
        cluster : bool, optional
            Whether to conduct cluster thresholding, for
            use when binary_importance="binary"

        Returns
        -------
        str, os.PathLike
            Location of generated mask

        """
        # Validate args

        def print_err(arg_name: str, arg_value: str):
            """Raise error for unsupported args."""
            raise ValueError(f"Unexpected {arg_name} value : {arg_value}")

        if task_name not in ["movies", "scenarios", "both"]:
            print_err("task_name", task_name)
        if model_name not in ["sep", "tog", "rest", "lss"]:
            print_err("model_name", model_name)
        if con_name not in ["stim", "tog", "replay"]:
            print_err("con_name", con_name)
        if binary_importance not in ["binary", "importance"]:
            print_err("binary_importance", binary_importance)
        if cluster and not binary_importance == "binary":
            raise ValueError(
                "Option 'cluster' only available when "
                + "binary_importance='binary'"
            )

        emo_list = self.emo_names()
        if emo_name not in emo_list:
            print_err("emo_name", emo_name)

        # Check for required attrs
        if not hasattr(self, "_empty_matrix") and not hasattr(
            self, "_img_header"
        ):
            raise AttributeError(
                "Attributes _empty_matrix, _img_header "
                + "required. Try ImportanceMask._mine_template."
            )

        # Get ID values
        db_con = sql_database.DbConnect()
        task_id, _ = db_con.fetch_rows(
            f"select * from ref_fsl_task where task_name = '{task_name}'"
        )[0]
        model_id, _ = db_con.fetch_rows(
            f"select * from ref_fsl_model where model_name = '{model_name}'"
        )[0]
        con_id, _ = db_con.fetch_rows(
            f"select * from ref_fsl_contrast where con_name = '{con_name}'"
        )[0]

        # Pull data
        plsda_table = f"tbl_plsda_{binary_importance}_gm"
        print(
            f"\tDownloading data from {plsda_table} for emotion : {emo_name}"
        )
        sql_cmd = f"""select distinct
            b.voxel_name, a.emo_{emo_name}
            from {plsda_table} a
            join ref_voxel_gm b on a.voxel_id = b.voxel_id
            where a.fsl_task_id = {task_id} and a.fsl_model_id = {model_id}
                and a.fsl_con_id = {con_id}
        """
        df_bin = db_con.fetch_df(sql_cmd, ["voxel_name", emo_name])
        db_con.close_con()

        # Build and write NIfTI file
        print(
            f"\tBuilding {binary_importance} map for emotion : {emo_name} ..."
        )
        out_file = (
            f"{binary_importance}_model-{model_name}_task-{task_name}_"
            + f"con-{con_name}_emo-{emo_name}_map.nii.gz"
        )
        out_path = os.path.join(out_dir, out_file)
        arr_fill = self._empty_matrix.copy()
        for _, row in df_bin.iterrows():
            x, y, z = row["voxel_name"].split(".")
            arr_fill[int(x)][int(y)][int(z)] = row[emo_name]
        emo_img = nib.Nifti1Image(
            arr_fill, affine=None, header=self._img_header
        )
        nib.save(emo_img, out_path)
        print(f"\tWrote : {out_path}")

        # Manage clustering
        if cluster:
            clust_size = 5 if task_name == "scenarios" else 10
            self.cluster(out_path, size=clust_size, vox_value=1)
        return out_path


class ConjunctAnalysis(_MapMethods):
    """Generate conjunction maps.

    Inherits _MapMethods.

    Generate omnibus, arousal, and valence conjunction maps
    from voxel importance maps.

    Parameters
    ----------
    map_list : list
        Paths to NIfTI voxel importance maps in template space
    out_dir : str, os.PathLike
        Output location

    Methods
    -------
    omni_map()
        Generate omnibus conjunction from all map_list files
    valence_map()
        Generate positive, negative, neutrual valence conjunction maps
    arousal_map()
        Generate high, medium, low arousal conjunction maps

    Example
    -------
    conj = fsl.group.ConjunctAnalysis(*args)
    conj.omni_map()
    conj.arousal_map()
    conj.valence_map()

    """

    def __init__(self, map_list, out_dir):
        """Initialize."""
        self._map_list = map_list
        self._out_dir = out_dir
        (
            self._model_level,
            self._model_name,
            self._task_name,
            self._con_name,
            _emo,
            _suff,
        ) = os.path.basename(map_list[0]).split("_")
        self._clust_size = 5 if self._task_name == "scenarios" else 10

    def omni_map(self):
        """Generate omnibus conjunction from all map_list files."""
        omni_out = os.path.join(
            self._out_dir,
            f"{self._model_level}_{self._model_name}_{self._task_name}_"
            + f"{self._con_name}_conj-omni_map.nii.gz",
        )
        print("Building conjunction map : omni")
        self.c3d_add(self._map_list, omni_out)
        self.cluster(omni_out, size=self._clust_size, vox_value=1)

    def valence_map(self):
        """Generate positive, negative, neutrual valence conjunction maps."""
        map_val = {
            "Pos": ["amusement", "awe", "excitement", "joy", "romance"],
            "Neg": [
                "anger",
                "anxiety",
                "disgust",
                "fear",
                "horror",
                "sadness",
            ],
            "Neu": ["calmness", "craving", "neutral", "surprise"],
        }
        self._conj_aro_val(map_val, "val")

    def _conj_aro_val(self, map_dict: dict, conj_name: str):
        """Unpack filenames to list items to make conj maps."""
        for key in map_dict:
            val_list = [
                x for x in self._map_list for y in map_dict[key] if y in x
            ]
            out_path = os.path.join(
                self._out_dir,
                f"{self._model_level}_{self._model_name}_{self._task_name}_"
                + f"{self._con_name}_conj-{conj_name}{key}_map.nii.gz",
            )
            print(f"Building conjunction map : {conj_name}{key}")
            self.c3d_add(val_list, out_path)
            self.cluster(out_path, size=self._clust_size, vox_value=1)

    def arousal_map(self):
        """Generate high, medium, low arousal conjunction maps."""
        map_aro = {
            "High": [
                "amusement",
                "anger",
                "anxiety",
                "craving",
                "disgust",
                "excitement",
                "fear",
                "horror",
                "surprise",
            ],
            "Low": ["calmness", "neutral", "sadness"],
            "Med": ["awe", "romance", "joy"],
        }
        self._conj_aro_val(map_aro, "aro")


# %%
