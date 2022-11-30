"""Pipelines supporting AFNI and FSL."""
# %%
import os
import glob
import pandas as pd
import numpy as np
from func_model import afni


# %%
def pipeline_afni(
    subj, proj_rawdata, proj_deriv, work_deriv, sing_afni, log_dir
):
    """Title.

    Desc.

    Sanity check - model processing during movie|scenario presenation.

    Parameters
    ----------
    subj
    proj_rawdata
    proj_deriv
    work_deriv
    sing_afni
    log_dir

    Raises
    ------
    FileNotFoundError
    ValueError

    """
    # Make timing files for each session
    sess_list = ["ses-day2", "ses-day3"]
    for sess in sess_list:

        # Check that session exists for participant
        subj_sess_raw = os.path.join(proj_rawdata, subj, sess)
        if not os.path.exists(subj_sess_raw):
            print(f"Directory not detected : {subj_sess_raw}\n\tSkipping.")

        # Setup output directory
        subj_work = os.path.join(work_deriv, "model_afni", subj, sess, "func")
        if not os.path.exists(subj_work):
            os.makedirs(subj_work)

        # Find events files
        sess_events = sorted(glob.glob(f"{subj_sess_raw}/func/*events.tsv"))
        if not sess_events:
            raise FileNotFoundError(
                f"Expected BIDs events files in {subj_sess_raw}"
            )

        # Identify and validate task name
        task = (
            os.path.basename(sess_events[0]).split("task-")[-1].split("_")[0]
        )
        task_valid = ["movies", "scenarios"]
        if task not in task_valid:
            raise ValueError(
                f"Expected task names movies|scenarios, found {task}"
            )

        # Generate timing files
        make_tf = afni.TimingFiles(subj, sess, task, subj_work, sess_events)
        make_tf.common_events()
        make_tf.task_events()


    # Generate deconvolution matrics

    # Run REML


def pipeline_fsl():
    """Title.

    Desc.

    """
    pass
