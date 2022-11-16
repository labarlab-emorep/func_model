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

    """
    #
    sess_list = ["ses-day2", "ses-day3"]
    subj_work = os.path.join(work_deriv, "model_afni", subj, sess, "func")
    if not os.path.exists(subj_work):
        os.makedirs(subj_work)

    # Generate timing files
    # TODO check number of events = 8
    # TODO check task == movies|scenarios
    sess = sess_list[0]
    subj_sess_raw = os.path.join(proj_rawdata, subj, sess)
    sess_events = sorted(glob.glob(f"{subj_sess_raw}/func/*events.tsv"))
    task = os.path.basename(sess_events[0]).split("task-")[-1].split("_")[0]


    # Generate deconvolution matrics

    # Run REML


def pipeline_fsl():
    """Title.

    Desc.

    """
    pass
