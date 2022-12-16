"""Run processing methods for workflow."""
# %%
import os
import glob
import fnmatch
from func_model import afni


# %%
def afni_sanity_tfs(subj, sess, subj_work, subj_sess_raw):
    """Make timing files for sanity check.

    Generate a set of AFNI-styled timing files in order to
    check the design and manipulation.

    Timing files for common, session stimulus, and selection
    tasks are generated. Fixations are excluded to serve as
    model baseline.

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    subj_work : path
        Location of working directory for generating intermediate files
    subj_sess_raw : path
        Location of participant's session rawdata, used to find
        BIDS event files

    Returns
    -------
    dict
        key = indicate type of timing file
        value = list of paths to timing files

    Raises
    ------
    FileNotFoundError
        BIDS event files are missing
    ValueError
        Unexpected task name

    """
    # Find events files
    sess_events = sorted(glob.glob(f"{subj_sess_raw}/func/*events.tsv"))
    if not sess_events:
        raise FileNotFoundError(
            f"Expected BIDs events files in {subj_sess_raw}"
        )

    # Identify and validate task name
    task = os.path.basename(sess_events[0]).split("task-")[-1].split("_")[0]
    task_valid = ["movies", "scenarios"]
    if task not in task_valid:
        raise ValueError(f"Expected task names movies|scenarios, found {task}")

    # Generate timing files
    sess_tfs = {}
    make_tf = afni.TimingFiles(subj, sess, task, subj_work, sess_events)
    sess_tfs["common"] = make_tf.common_events()
    sess_tfs["session"] = make_tf.session_events()
    sess_tfs["select"] = make_tf.select_events()
    return sess_tfs


# %%
def afni_sanity_preproc(subj, sess, subj_work, proj_deriv, sing_afni):
    """Title.

    Desc.

    Parameters
    ----------
    subj
    sess
    subj_work
    proj_deriv
    sing_afni

    Returns
    -------
    tuple

    Raises
    ------
    FileNotFoundError

    """
    # Set search dictionary used to make anat_dict
    #   key = identifier of file in anat_dict
    #   value = searchable string by glob
    get_anat_dict = {
        "anat-preproc": "desc-preproc_T1w",
        "mask-brain": "desc-brain_mask",
        "mask-probCS": "label-CSF_probseg",
        "mask-probGM": "label-GM_probseg",
        "mask-probWM": "label-WM_probseg",
    }

    # Setup dictionary of anatomical files
    anat_dict = {}
    subj_deriv_fp = os.path.join(proj_deriv, f"pre_processing/fmriprep/{subj}")
    for key, search in get_anat_dict.items():
        file_path = sorted(
            glob.glob(
                f"{subj_deriv_fp}/**/anat/{subj}_*space-*_{search}.nii.gz",
                recursive=True,
            )
        )
        if file_path:
            anat_dict[key] = file_path[0]
        else:
            raise FileNotFoundError(
                f"Expected to find fmriprep file anat/*_{search}.nii.gz"
            )

    #
    func_dict = {}
    mot_files = [
        x
        for x in glob.glob(f"{subj_deriv_fp}/{sess}/func/*timeseries.tsv")
        if not fnmatch.fnmatch(x, "*task-rest*")
    ]
    mot_files.sort()
    if not mot_files:
        raise FileNotFoundError(
            "Expected to find fmriprep files func/*timeseries.tsv"
        )

    # Get AROMA files, which are spatially smoothed
    subj_deriv_fsl = os.path.join(
        proj_deriv, f"pre_processing/fsl_denoise/{subj}/{sess}/func"
    )
    run_files = [
        x
        for x in glob.glob(f"{subj_deriv_fsl}/*tfiltMasked_bold.nii.gz")
        if not fnmatch.fnmatch(x, "*task-rest*")
    ]
    run_files.sort()
    if not run_files:
        raise FileNotFoundError(
            "Expected to find fsl_denoise files *tfiltMasked_bold.nii.gz"
        )

    if len(run_files) != len(mot_files):
        raise ValueError(
            f"""
        Length of motion files differs from run files.
            motion files : {mot_files}
            run files : {run_files}
        """
        )

    func_dict = {}
    func_dict["func-motion"] = mot_files
    func_dict["func-preproc"] = run_files

    # make masks
    make_masks = afni.MakeMasks(
        subj, sess, subj_work, proj_deriv, anat_dict, func_dict, sing_afni
    )
    make_masks.intersect()
    make_masks.tissue()
    make_masks.minimum()
    anat_dict = make_masks.anat_dict
    del make_masks

    # smooth
    smooth_epi = afni.smooth_epi(
        subj_work, proj_deriv, func_dict["func-preproc"], sing_afni
    )

    # scale
    func_dict["func-scaled"] = afni.scale_epi(
        subj_work,
        proj_deriv,
        anat_dict["mask-minimum"],
        smooth_epi,
        sing_afni,
    )

    # make afni-style motion files
    return (func_dict, anat_dict)


# # %%
# def afni_sanity_motion():
#     """Title.

#     Desc.

#     """
#     pass
