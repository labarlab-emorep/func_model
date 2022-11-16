"""Methods for AFNI."""
import os
import pandas as pd
import numpy as np


def movie_timing(subj_work, sess_events):
    """Title.

    Desc.

    """
    # Read in events, add run info
    events_data = [pd.read_table(x) for x in sess_events]
    events_run = [int(x.split("_run-")[1].split("_")[0]) for x in sess_events]
    for idx, _ in enumerate(events_data):
        events_data[idx]["run"] = events_run[idx]
    df_events = pd.concat(events_data)

    # Use fix as baseline, model movie|scenario, judge, emotion,
    # intensity, and wash
    trial_types = df_events.trial_type.unique()
    trial_types = ["fix", "replay", "judge", "emotion", "intensity", "wash"]

    # find movies, get unique stimulus type
    idx_movie = df_events.index[df_events["trial_type"] == "movie"].tolist()
    stim_info_list = df_events.loc[idx_movie, "stim_info"].tolist()
    emo_all_list = [x.split("_")[0] for x in stim_info_list]
    emo_list = np.unique(np.array(emo_all_list)).tolist()

    # Start empty TF
    subj_tf_dir = os.path.join(subj_work, "timing_files")
    if not os.path.exists(subj_tf_dir):
        os.makedirs(subj_tf_dir)


def scenario_timing():
    """Title.

    Desc.

    """
    pass
