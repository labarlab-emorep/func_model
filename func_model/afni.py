"""Methods for AFNI."""
import os
import pandas as pd
import numpy as np


def movie_timing(subj, sess, task, subj_work, sess_events):
    """Title.

    Desc.

    """
    # Read in events, add run info. Use run value from file
    # name string to anticipate missing runs.
    # TODO check that events_data, events_run have same length
    events_data = [pd.read_table(x) for x in sess_events]
    events_run = [int(x.split("_run-")[1].split("_")[0]) for x in sess_events]
    for idx, _ in enumerate(events_data):
        events_data[idx]["run"] = events_run[idx]
    df_events = pd.concat(events_data).reset_index(drop=True)

    #
    subj_tf_dir = os.path.join(subj_work, "timing_files")
    if not os.path.exists(subj_tf_dir):
        os.makedirs(subj_tf_dir)

    # Set basic trial types -- fix will be baseline, so do not include.
    # TODO deal with different emotion responses and NRs
    common_switch = {
        "replay": "evRep",
        "judge": "evJud",
        "emotion": "evEmo",
        "intensity": "evInt",
        "wash": "evWas",
    }

    #
    for event, tf_name in common_switch.items():

        # Make empty file
        tf_path = os.path.join(
            subj_tf_dir, f"{subj}_{sess}_task-{task}_desc-{tf_name}_events.1D"
        )
        open(tf_path, "w").close()

        #
        chk_run = 0
        for run in events_run:

            #
            idx_event = df_events.index[
                (df_events["trial_type"] == event) & (df_events["run"] == run)
            ].tolist()
            onset = df_events.loc[idx_event, "onset"].tolist()
            duration = df_events.loc[idx_event, "duration"].tolist()
            ons_dur = [f"{ons}:{dur}" for ons, dur in zip(onset, duration)]

            # Anticipate missing runs, e.g. ER0093 where chk_run == 6,
            # events_run ==7. If a run is missing, use AFNI fill
            # and then write events_run to a new line. Update
            # chk_run to realign.
            chk_run += 1
            if run == chk_run:
                line_content = " ".join(ons_dur)
            else:
                line_content = f"*\n{' '.join(ons_dur)}"
                chk_run += 1

            #
            with open(tf_path, "a") as tf:
                tf.writelines(f"{line_content}\n")

    # Find movie trials, determine emotion trial types
    idx_movie = df_events.index[df_events["trial_type"] == "movie"].tolist()
    stim_info_list = df_events.loc[idx_movie, "stim_info"].tolist()
    emo_all_list = [x.split("_")[0] for x in stim_info_list]
    emo_list = np.unique(np.array(emo_all_list)).tolist()

    #
    # TODO check keys are in emo_list
    movie_switch = {
        "amusement": "mvAmu",
        "anger": "mvAng",
        "anxiety": "mvAnx",
        "awe": "mvAwe",
        "calmness": "mvCal",
        "craving": "mvCra",
        "disgust": "mvDis",
        "excitement": "mvExc",
        "fear": "mvFea",
        "horror": "mvHor",
        "joy": "mvJoy",
        "neutral": "mvNeu",
        "romance": "mvRom",
        "sadness": "mvSad",
        "surprise": "mvSur",
    }

    #
    for emo, tf_name in movie_switch.items():

        # Make empty file
        tf_path = os.path.join(
            subj_tf_dir, f"{subj}_{sess}_task-{task}_desc-{tf_name}_events.1D"
        )
        open(tf_path, "w").close()

        #
        chk_run = 0
        for run in events_run:

            #
            idx_emo = df_events.index[
                (df_events["stim_info"].str.contains(emo))
                & (df_events["run"] == run)
            ].tolist()

            #
            if not idx_emo:
                h_line = "*\n"
            else:
                onset = df_events.loc[idx_emo, "onset"].tolist()
                duration = df_events.loc[idx_emo, "duration"].tolist()
                ons_dur = [f"{ons}:{dur}" for ons, dur in zip(onset, duration)]
                h_line = f"{' '.join(ons_dur)}\n"

            #
            chk_run += 1
            if run == chk_run:
                line_content = h_line
            else:
                line_content = f"*\n{h_line}"
                chk_run += 1

            #
            with open(tf_path, "a") as tf:
                tf.writelines(f"{line_content}")


def scenario_timing():
    """Title.

    Desc.

    """
    pass
