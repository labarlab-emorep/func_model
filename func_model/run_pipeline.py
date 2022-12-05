"""Run processing methods for workflow."""
import os
import glob
import fnmatch
from func_model import afni


# %%
def afni_sanity_tfs(subj, sess, subj_work, subj_sess_raw):
    """Title.

    Desc.

    Parameters
    ----------
    subj
    sess
    subj_work
    subj_sess_raw

    Returns
    -------
    dict

    Raises
    ------

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
def afni_sanity_preproc(subj, sess, subj_work, proj_deriv):
    """Title.

    Desc.

    """
    #
    get_anat_dict = {
        "anat-preproc": "desc-preproc_T1w",
        "mask-brain": "desc-brain_mask",
        "mask-probCS": "label-CSF_probseg",
        "mask-probGM": "label-GM_probseg",
        "mask-probWM": "label-WM_progseg",
    }

    #
    anat_dict = {}
    subj_deriv_fp = os.path.join(proj_deriv, f"pre_processing/fmriprep/{subj}")
    for key, search in get_anat_dict.items():
        file_path = sorted(
            glob.glob(
                f"{subj_deriv_fp}/**/anat/{subj}_*{search}.nii.gz",
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
        if not fnmatch.fnmatch("task-rest", x)
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
        for x in glob.glob(f"{subj_deriv_fsl}/*tfiltAROMAMasked_bold.nii.gz")
        if not fnmatch.fnmatch("task-rest", x)
    ]
    run_files.sort()
    if not mot_files:
        raise FileNotFoundError(
            "Expected to find fsl_denoise files *tfiltAROMAMasked_bold.nii.gz"
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
    make_masks = afni.MakeMasks(subj, sess, anat_dict, func_dict)
    make_masks.intersect()

    # scale

    # make afni-style motion files


# # %%
# def afni_sanity_motion():
#     """Title.

#     Desc.

#     """
#     pass


# # %%
# def afni_sanity_masks():
#     """Title.

#     Desc.

#     """
#     pass
