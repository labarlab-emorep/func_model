"""Methods for AFNI."""
import os
import pandas as pd
import numpy as np


class TimingFiles:
    """Title.

    Desc.

    """

    def __init__(self, subj, sess, task, subj_work, sess_events):
        """Title.

        Desc.

        Attributes
        ----------
        subj
        sess
        task
        sess_events
        subj_tf_dir

        Raises
        ------
        ValueError

        """
        # Check arguments
        task_valid = ["movies", "scenarios"]
        if task not in task_valid:
            raise ValueError(
                f"Expected task names movies|scenarios, found {task}"
            )
        if len(sess_events) < 1:
            raise ValueError("Cannot make timing files from 0 events.tsv")

        # Set attributes
        self.subj = subj
        self.sess = sess
        self.task = task
        self.sess_events = sess_events
        self.subj_tf_dir = os.path.join(subj_work, "timing_files")
        if not os.path.exists(self.subj_tf_dir):
            os.makedirs(self.subj_tf_dir)

        # Generate dataframe from events files
        self._event_dataframe()

    def _event_dataframe(self):
        """Title.

        Desc.

        Attributes
        ----------
        df_events
        events_run

        """
        # Read-in events files, construct list of dataframes. Determine
        # run info from file name.
        events_data = [pd.read_table(x) for x in self.sess_events]
        self.events_run = [
            int(x.split("_run-")[1].split("_")[0]) for x in self.sess_events
        ]
        if len(events_data) != len(self.events_run):
            raise ValueError("Number of runs and events files differ")

        # Add run info to listed dataframes, construct session dataframe
        for idx, _ in enumerate(events_data):
            events_data[idx]["run"] = self.events_run[idx]
        self.df_events = pd.concat(events_data).reset_index(drop=True)

    def _check_run(self, current_run, count_run, line_content):
        """Title.

        # Anticipate missing runs, e.g. ER0093 where chk_run == 6,
        # events_run ==7.

        Parameters
        ----------
        current_run
        count_run
        line_content

        Returns
        -------
        tuple

        """
        # Prepend an AFNI empty line if counter is ahead of
        # run, update counter.
        if current_run == count_run:
            return (line_content, count_run)
        else:
            line_fill = f"*\n{line_content}"
            count_run += 1
            return (line_fill, count_run)

    def _onset_duration(self, idx_event, marry):
        """Title.

        Desc.

        Parameters
        ----------
        idx_event
        marry

        Returns
        -------
        list

        """
        #
        onset = self.df_events.loc[idx_event, "onset"].tolist()
        duration = self.df_events.loc[idx_event, "duration"].tolist()
        if marry:
            return [f"{ons}:{dur}" for ons, dur in zip(onset, duration)]
        else:
            return [str(x) for x in onset]

    def common_events(self, marry=True, event_name=None):
        """Title.

        Desc.

        Parameters
        ----------
        marry
        event_name

        Notes
        -----

        """
        #
        if not isinstance(marry, bool):
            raise TypeError("Argument 'marry' is bool")

        # Set basic trial types -- fix will be baseline, so do not include.
        # TODO deal with different emotion responses and NRs
        common_dict = {
            "replay": "comRep",
            "judge": "comJud",
            "emotion": "comEmo",
            "intensity": "comInt",
            "wash": "comWas",
        }

        #
        if event_name:
            valid_list = [x for x in common_dict.keys()]
            if event_name not in valid_list:
                raise ValueError(
                    f"Inappropriate event supplied : {event_name}"
                )
            h_dict = {}
            h_dict[event_name] = common_dict[event_name]
            del common_dict
            common_dict = h_dict

        #
        for event, tf_name in common_dict.items():

            # Make empty file
            tf_path = os.path.join(
                self.subj_tf_dir,
                f"{self.subj}_{self.sess}_task-{self.task}_"
                + f"desc-{tf_name}_events.1D",
            )
            open(tf_path, "w").close()

            #
            chk_run = 0
            for run in self.events_run:

                #
                idx_event = self.df_events.index[
                    (self.df_events["trial_type"] == event)
                    & (self.df_events["run"] == run)
                ].tolist()
                ons_dur = self._onset_duration(idx_event, marry)
                h_line = " ".join(ons_dur)

                #
                chk_run += 1
                line_content, chk_run = self._check_run(run, chk_run, h_line)

                #
                with open(tf_path, "a") as tf:
                    tf.writelines(f"{line_content}\n")

    def task_events(self, marry=True, emotion_name=None):
        """Title.

        Desc.

        Parameters
        ----------
        marry
        emotion_name

        Notes
        -----

        """
        #
        if not isinstance(marry, bool):
            raise TypeError("Argument 'marry' is bool")

        #
        get_events = getattr(self, f"_{self.task}_events")
        get_events(marry, emotion_name)

    def _movies_events(self, marry, emotion_name):
        """Title.

        Desc.

        Parameters
        ----------
        marry
        emotion_name

        Notes
        -----

        """
        #
        movie_dict = {
            "amusement": "movAmu",
            "anger": "movAng",
            "anxiety": "movAnx",
            "awe": "movAwe",
            "calmness": "movCal",
            "craving": "movCra",
            "disgust": "movDis",
            "excitement": "movExc",
            "fear": "movFea",
            "horror": "movHor",
            "joy": "movJoy",
            "neutral": "movNeu",
            "romance": "movRom",
            "sadness": "movSad",
            "surprise": "movSur",
        }

        # Find movie trials, determine emotion trial types
        if emotion_name:
            valid_list = [x for x in movie_dict.keys()]
            if emotion_name not in valid_list:
                raise ValueError(
                    f"Inappropriate emotion supplied : {emotion_name}"
                )
            emo_list = [emotion_name]
        else:
            idx_movie = self.df_events.index[
                self.df_events["trial_type"] == "movie"
            ].tolist()
            emo_all = self.df_events.loc[idx_movie, "emotion"].tolist()
            emo_list = np.unique(np.array(emo_all)).tolist()

        #
        for emo in emo_list:

            #
            if emo not in movie_dict.keys():
                raise ValueError(f"Unexpected emotion encountered : {emo}")

            # Make empty file
            tf_name = movie_dict[emo]
            tf_path = os.path.join(
                self.subj_tf_dir,
                f"{self.subj}_{self.sess}_task-{self.task}_"
                + f"desc-{tf_name}_events.1D",
            )
            open(tf_path, "w").close()

            #
            chk_run = 0
            for run in self.events_run:

                #
                idx_emo = self.df_events.index[
                    (self.df_events["emotion"] == emo)
                    & (self.df_events["run"] == run)
                ].tolist()

                #
                if not idx_emo:
                    h_line = "*"
                else:
                    ons_dur = self._onset_duration(idx_emo, marry)
                    h_line = " ".join(ons_dur)

                #
                chk_run += 1
                line_content, chk_run = self._check_run(run, chk_run, h_line)

                #
                with open(tf_path, "a") as tf:
                    tf.writelines(f"{line_content}\n")


# def movie_timing(subj, sess, task, subj_work, sess_events):
#     """Title.

#     Desc.

#     """
#     # Read in events, add run info. Use run value from file
#     # name string to anticipate missing runs.
#     # TODO check that events_data, events_run have same length
#     events_data = [pd.read_table(x) for x in sess_events]
#     events_run = [int(x.split("_run-")[1].split("_")[0]) for x in sess_events]
#     for idx, _ in enumerate(events_data):
#         events_data[idx]["run"] = events_run[idx]
#     df_events = pd.concat(events_data).reset_index(drop=True)

#     #
#     subj_tf_dir = os.path.join(subj_work, "timing_files")
#     if not os.path.exists(subj_tf_dir):
#         os.makedirs(subj_tf_dir)

#     # Set basic trial types -- fix will be baseline, so do not include.
#     # TODO deal with different emotion responses and NRs
#     common_switch = {
#         "replay": "evRep",
#         "judge": "evJud",
#         "emotion": "evEmo",
#         "intensity": "evInt",
#         "wash": "evWas",
#     }

#     #
#     for event, tf_name in common_switch.items():

#         # Make empty file
#         tf_path = os.path.join(
#             subj_tf_dir, f"{subj}_{sess}_task-{task}_desc-{tf_name}_events.1D"
#         )
#         open(tf_path, "w").close()

#         #
#         chk_run = 0
#         for run in events_run:

#             #
#             idx_event = df_events.index[
#                 (df_events["trial_type"] == event) & (df_events["run"] == run)
#             ].tolist()
#             onset = df_events.loc[idx_event, "onset"].tolist()
#             duration = df_events.loc[idx_event, "duration"].tolist()
#             ons_dur = [f"{ons}:{dur}" for ons, dur in zip(onset, duration)]

#             # Anticipate missing runs, e.g. ER0093 where chk_run == 6,
#             # events_run ==7. If a run is missing, use AFNI fill
#             # and then write events_run to a new line. Update
#             # chk_run to realign.
#             chk_run += 1
#             if run == chk_run:
#                 line_content = " ".join(ons_dur)
#             else:
#                 line_content = f"*\n{' '.join(ons_dur)}"
#                 chk_run += 1

#             #
#             with open(tf_path, "a") as tf:
#                 tf.writelines(f"{line_content}\n")

#     # Find movie trials, determine emotion trial types
#     idx_movie = df_events.index[df_events["trial_type"] == "movie"].tolist()
#     stim_info_list = df_events.loc[idx_movie, "stim_info"].tolist()
#     emo_all_list = [x.split("_")[0] for x in stim_info_list]
#     emo_list = np.unique(np.array(emo_all_list)).tolist()

#     #
#     # TODO check keys are in emo_list
#     movie_switch = {
#         "amusement": "movAmu",
#         "anger": "movAng",
#         "anxiety": "movAnx",
#         "awe": "movAwe",
#         "calmness": "movCal",
#         "craving": "movCra",
#         "disgust": "movDis",
#         "excitement": "movExc",
#         "fear": "movFea",
#         "horror": "movHor",
#         "joy": "movJoy",
#         "neutral": "movNeu",
#         "romance": "movRom",
#         "sadness": "movSad",
#         "surprise": "movSur",
#     }

#     #
#     for emo, tf_name in movie_switch.items():

#         # Make empty file
#         tf_path = os.path.join(
#             subj_tf_dir, f"{subj}_{sess}_task-{task}_desc-{tf_name}_events.1D"
#         )
#         open(tf_path, "w").close()

#         #
#         chk_run = 0
#         for run in events_run:

#             #
#             idx_emo = df_events.index[
#                 (df_events["stim_info"].str.contains(emo))
#                 & (df_events["run"] == run)
#             ].tolist()

#             #
#             if not idx_emo:
#                 h_line = "*\n"
#             else:
#                 onset = df_events.loc[idx_emo, "onset"].tolist()
#                 duration = df_events.loc[idx_emo, "duration"].tolist()
#                 ons_dur = [f"{ons}:{dur}" for ons, dur in zip(onset, duration)]
#                 h_line = f"{' '.join(ons_dur)}\n"

#             #
#             chk_run += 1
#             if run == chk_run:
#                 line_content = h_line
#             else:
#                 line_content = f"*\n{h_line}"
#                 chk_run += 1

#             #
#             with open(tf_path, "a") as tf:
#                 tf.writelines(f"{line_content}")


def scenario_timing():
    """Title.

    Desc.

    """
    pass
