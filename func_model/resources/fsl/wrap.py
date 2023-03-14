"""Wrapper methods for running FSL workflows.

To offload workflows, coordinate pipeline methods and their
required inputs.

"""
# %%
import os
import glob
from typing import Union
import nibabel as nib
from . import model, helper


# %%
def make_condition_files(
    subj, sess, task, model_name, subj_work, proj_rawdata
):
    """Make condition files from BIDS events files.

    Identify required files and wrap class fsl.model.ConditionFiles. Use
    model_name to trigger appropriate class methods for making desired
    condition files.

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    task : str
        BIDS task identifier
    model_name : str
        Name of FSL model
    subj_work : path
        Output work location for intermediates
    proj_rawdata : path
        Location of BIDS rawdata

    Raises
    ------
    FileNotFoundError
        Missing expected BIDS events files for subj, sess
    ValueError
        Unexpected model_name

    """
    if not helper.valid_name(model_name):
        raise ValueError(f"Improper model name : {model_name}")

    # Find BIDS events files
    subj_sess_raw = os.path.join(proj_rawdata, subj, sess, "func")
    sess_events = sorted(glob.glob(f"{subj_sess_raw}/*events.tsv"))
    if not sess_events:
        raise FileNotFoundError(
            f"Expected BIDs events files in {subj_sess_raw}"
        )

    # Make condition files for each session run and event
    make_cf = model.ConditionFiles(subj, sess, task, subj_work, sess_events)
    for run_num in make_cf.run_list:
        make_cf.common_events(run_num)
        if model_name == "sep":
            make_cf.session_separate_events(run_num)


def make_confound_files(subj, sess, task, subj_work, proj_deriv):
    """Make confounds files from fMRIPrep timeseries files.

    Identify required files and wrap function fsl.model.confounds.

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    task : str
        BIDS task identifier
    subj_work : path
        Output work location for intermediates
    proj_deriv : path
        Location of project BIDs derivatives, for finding
        fMRIPrep output

    Raises
    ------
    FileNotFoundError
        Failed to find any fMRIPrep confound files

    """
    # Find fMRIPrep confounds
    fp_subj_sess = os.path.join(
        proj_deriv, "pre_processing/fmriprep", subj, sess
    )
    sess_confounds = sorted(
        glob.glob(f"{fp_subj_sess}/func/*{task}*timeseries.tsv")
    )
    if not sess_confounds:
        raise FileNotFoundError(
            f"Expected fMRIPrep confounds files in {fp_subj_sess}"
        )

    # Make confound files
    for conf_path in sess_confounds:
        _ = model.confounds(conf_path, subj_work, fd_thresh=0.5)


def write_first_fsf(subj, sess, task, model_name, subj_work, proj_deriv):
    """Write first-level FSF design files.

    Identify required files and wrap class fsl.model.MakeFirstFsf. Requires
    output of model.ConditionFiles, model.confounds.

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    task : str
        BIDS task identifier
    model_name : str
        Name of FSL model
    subj_work : path
        Output work location for intermediates
    proj_deriv : path
        Location of project BIDs derivatives, for finding
        fsl_denoise output

    Returns
    -------
    list
        Paths to generated design files

    Raises
    ------
    FileNotFoundError
        Missing FSL preprocessed files
        Missing regressor files
    ValueError
        Unexpected BIDS run number

    """

    def _get_run(file_name: str) -> str:
        "Return run field from temporal filtered filename."
        try:
            _su, _se, _ta, run, _sp, _re, _de, _su = file_name.split("_")
            return run
        except IndexError:
            raise ValueError(
                "Improperly formatted file name for preprocessed BOLD."
            )

    def _get_file(search_path: str, run: str, desc: str) -> Union[str, None]:
        """Return path to condition/confound file or None."""
        try:
            return glob.glob(f"{search_path}/*_{run}_{desc}*.txt")[0]
        except IndexError:
            return None

    def _get_cond(search_path: str, run: str) -> dict:
        """Return dict of paths to common conditions."""
        # cond_common values match description field of condition files
        cond_common = ["judgment", "washout", "emoSelect", "emoIntensity"]
        out_dict = {}
        for cond in cond_common:
            out_dict[cond] = _get_file(search_path, run, f"desc-{cond}")
        return out_dict

    def _none_in_dict(search_dict: dict) -> bool:
        """Check dictionary values for None types."""
        for _key, value in search_dict.items():
            if value is None:
                return True
        return False

    # Identify preprocessed FSL files
    fd_subj_sess = os.path.join(
        proj_deriv, "pre_processing/fsl_denoise", subj, sess
    )
    sess_preproc = sorted(
        glob.glob(f"{fd_subj_sess}/func/*{task}*desc-scaled_bold.nii.gz")
    )
    if not sess_preproc:
        raise FileNotFoundError(
            f"Expected fsl_denoise files in {fd_subj_sess}"
        )

    # Make run-specific design files
    make_fsf = model.MakeFirstFsf(subj_work, proj_deriv, model_name)
    fsf_list = []
    for preproc_path in sess_preproc:

        # Determine number of volumes
        img = nib.load(preproc_path)
        img_header = img.header
        num_vol = img_header.get_data_shape()[3]
        del img

        # Find confounds file - failing to find a confounds may be
        # due to excessive motion detected, whether or not to model
        # a run is being decided by whether a confounds file exists.
        # See resources.fsl.model.confounds for more details.
        run = _get_run(os.path.basename(preproc_path))
        search_conf = f"{subj_work}/confounds_files"
        confound_path = _get_file(search_conf, run, "desc-confounds")
        if not confound_path:
            print(f"\tNo confound found for {run}, skipping")
            continue

        # Find condition files
        search_cond = f"{subj_work}/condition_files"
        cond_dict = _get_cond(search_cond, run)
        if _none_in_dict(cond_dict):
            print(f"\tMissing required condition file for {run}, skipping")
            continue

        # Write design file
        use_short = True if run == "run-04" or run == "run-08" else False
        fsf_path = make_fsf.write_fsf(
            run,
            num_vol,
            preproc_path,
            confound_path,
            cond_dict["judgment"],
            cond_dict["washout"],
            cond_dict["emoSelect"],
            cond_dict["emoIntensity"],
            use_short,
        )
        fsf_list.append(fsf_path)
    return fsf_list
