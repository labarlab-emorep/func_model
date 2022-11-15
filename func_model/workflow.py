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

    """
    sess_list = ["ses-day2", "ses-day3"]

    # Generate timing files
    # TODO check number of events = 8
    # TODO check task == movies|scenarios
    sess = sess_list[0]
    subj_sess_raw = os.path.join(proj_rawdata, subj, sess)
    sess_events = sorted(glob.glob(f"{subj_sess_raw}/func/*events.tsv"))
    task = os.path.basename(sess_events[0]).split("task-")[-1].split("_")[0]

    # Read in events, add run info
    # TODO use run info from file string
    events_data = [pd.read_table(x) for x in sess_events]
    for idx, _ in enumerate(events_data):
        events_data[idx]["run"] = idx + 1
    df_events = pd.concat(events_data)

    # Use fix as baseline, model movie|scenario, judge, emotion,
    # intensity, and wash
    trial_types = df_events.trial_type.unique()

    # find movies, get unique stimulus type
    idx_movie = df_events.index[df_events["trial_type"] == "movie"].tolist()
    stim_info_list = df_events.loc[idx_movie, "stim_info"].tolist()
    stim_all_list = [x.split("_")[0] for x in stim_info_list]
    stim_list = np.unique(np.array(stim_all_list)).tolist()

    # Generate deconvolution matrics

    # Run REML


def pipeline_fsl():
    """Title.

    Desc.

    """
    pass
