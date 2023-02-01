"""Wrapper methods for running FSL workflows.

To offload workflows, coordinate pipeline methods and their
required inputs.

"""
# %%
import os
import glob
import nibabel as nib
from . import model


# %%
def make_condition_files(
    subj, sess, task, model_name, subj_work, proj_rawdata
):
    """Make condition files from BIDS events files.

    Identify required files and wrap fsl.model.ConditionFiles.

    Parameters
    ----------
    subj
    sess
    task
    model_name
    subj_work
    proj_rawdata

    Raises
    ------
    FileNotFoundError

    """
    #
    subj_sess_raw = os.path.join(proj_rawdata, subj, sess)
    sess_events = sorted(glob.glob(f"{subj_sess_raw}/func/*events.tsv"))
    if not sess_events:
        raise FileNotFoundError(
            f"Expected BIDs events files in {subj_sess_raw}"
        )

    #
    make_cf = model.ConditionFiles(subj, sess, task, subj_work, sess_events)
    for run_num in make_cf.run_list:
        make_cf.common_events(run_num)
        if model_name == "sep":
            make_cf.session_separate_events(run_num)


def make_confound_files(subj, sess, task, subj_work, proj_deriv):
    """Make confounds files from fMRIPrep timeseries files.

    Identify required files and wrap fsl.model.confounds.

    Parameters
    ----------
    subj
    sess
    task
    subj_work
    proj_deriv

    Raises
    ------
    FileNotFoundError

    """
    #
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

    #
    for conf_path in sess_confounds:
        _ = model.confounds(conf_path, subj_work)


def write_first_fsf(
    subj, sess, task, model_name, model_level, subj_work, proj_deriv
):
    """Write first-level FSF design files.

    Identify required files and wrap fsl.model.MakeFirstFsf.

    Parameters
    ----------
    subj
    sess
    task
    model_name
    model_level
    subj_work
    proj_deriv

    Returns
    -------
    list

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

    def _get_file(search_path: str, run: str, desc: str) -> str:
        """Return path to condition or confound file."""
        try:
            return glob.glob(f"{search_path}/*_{run}_{desc}*.txt")[0]
        except IndexError:
            raise FileNotFoundError(
                f"Missing {run} {desc} file in {search_path}"
            )

    #
    fd_subj_sess = os.path.join(
        proj_deriv, "pre_processing/fsl_denoise", subj, sess
    )
    sess_preproc = sorted(
        glob.glob(f"{fd_subj_sess}/func/*{task}*tfiltMasked_bold.nii.gz")
    )
    if not sess_preproc:
        raise FileNotFoundError(
            f"Expected fsl_denoise files in {fd_subj_sess}"
        )

    #
    make_fsf = model.MakeFirstFsf(
        subj_work, proj_deriv, model_name, model_level
    )
    fsf_list = []
    for preproc_path in sess_preproc:

        #
        run = _get_run(os.path.basename(preproc_path))
        img = nib.load(preproc_path)
        img_header = img.header
        num_vol = img_header.get_data_shape()[3]
        del img

        #
        search_conf = f"{subj_work}/confounds_files"
        search_cond = f"{subj_work}/condition_files"
        confound_path = _get_file(search_conf, run, "desc-confounds")
        judge_path = _get_file(search_cond, run, "desc-judgment")
        wash_path = _get_file(search_cond, run, "desc-washout")
        emosel_path = _get_file(search_cond, run, "desc-emoSelect")
        emoint_path = _get_file(search_cond, run, "desc-emoIntensity")

        #
        use_short = True if run == "run-04" or run == "run-08" else False
        fsf_path = make_fsf.write_fsf(
            run,
            num_vol,
            preproc_path,
            confound_path,
            judge_path,
            wash_path,
            emosel_path,
            emoint_path,
            use_short,
        )
        fsf_list.append(fsf_path)
    return fsf_list
