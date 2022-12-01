"""Methods for AFNI."""
import os
import pandas as pd
import numpy as np


class TimingFiles:
    """Create timing files for various types of EmoRep events.

    Aggregate all BIDS task events files for a participant's session,
    and then generate AFNI-style timing files (row number == run number)
    with onset or onset:duration information for each line.

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    task : str
        [movies | scenarios]
        Name of task
    subj_work : path
        Location of working directory for intermediates
    sess_events : list
        Paths to subject, session BIDS events files

    Attributes
    ----------
    df_events : pd.DataFrame
        Combined events data into single dataframe
    events_run : list
        Run identifier extracted from event file name
    sess : str
        BIDS session identifier
    sess_events : list
        Paths to subject, session BIDS events files
    subj : str
        BIDS subject identifier
    subj_tf_dir : path
        Output location for writing subject, session
        timing files
    task : str
        [movies | scenarios]
        Name of task

    Methods
    -------
    common_events(marry=True, common_name=None)
        Generate timing files for common events
        (replay, judge, and wash)
    select_events(marry=True, select_name=None)
        Generate timing files for selection trials
        (emotion, intensity)
    session_events(marry=True, emotion_name=None, emo_query=False)
        Generate timing files for movie or scenario emotions

    Notes
    -----
    -   Timing files are written to self.subj_tf_dir.
    -   All timing files use a 6 character camel-case identifier in
        the description field of the BIDS file name, the first
        3 characters specify the type of event and last 3
        characters indicate the event name. 6 characters are used
        due to limitations of sub-brick label lengths.
        For example:
        -   desc-comWas = a common event, wash specifically
        -   desc-movFea = a movie event, with fear stimuli
        -   desc-sceNeu = a scenario event, with neutral stimuli
        -   desc-selInt = a selection event, with intensity prompt

    """

    def __init__(self, subj, sess, task, subj_work, sess_events):
        """Initialize object.

        Setup attributes, make timing file directory, and combine all
        events files into a single dataframe.

        Parameters
        ----------
        subj : str
            BIDS subject identifier
        sess : str
            BIDS session identifier
        task : str
            [movies | scenarios]
            Name of task
        subj_work : path
            Location of working directory for intermediates
        sess_events : list
            Paths to subject, session BIDS events files

        Attributes
        ----------
        sess : str
            BIDS session identifier
        sess_events : list
            Paths to subject, session BIDS events files
        subj : str
            BIDS subject identifier
        subj_tf_dir : path
            Output location for writing subject, session
            timing files
        task : str
            [movies | scenarios]
            Name of task

        Raises
        ------
        ValueError
            Unexpected task name
            Insufficient number of sess_events

        """
        # Check arguments
        task_valid = ["movies", "scenarios"]
        if task not in task_valid:
            raise ValueError(
                f"Expected task names movies|scenarios, found {task}"
            )
        if len(sess_events) < 1:
            raise ValueError("Cannot make timing files from 0 events.tsv")

        # Set attributes, make output location
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
        """Combine data from events files into dataframe.

        Attributes
        ----------
        df_events : pd.DataFrame
            Column names == events files, run column added
        events_run : list
            Run identifier extracted from event file name

        Raises
        ------
        ValueError
            The number of events files and number of runs are unequal

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
        """Manage cases of missing run data.

        Deprecated.

        Anticipate missing run data, e.g. ER0093 who is missing
        run-6 events data. If current_run does not equal count_run
        (current_run = 7, count_run = 6 for ER0093 case), then
        prepend an AFNI-style empty line.

        Parameters
        ----------
        current_run : int
            Derived from self.events_run
        count_run : int
            Counter to track iteration number
        line_content : str
            Values to be added to AFNI-style timing file line

        Returns
        -------
        tuple
            [0] = original|updated line
            [1] = original|updated iteration counter

        """
        if current_run == count_run:
            return (line_content, count_run)
        else:
            line_fill = f"*\n{line_content}"
            count_run += 1
            return (line_fill, count_run)

    def _onset_duration(self, idx_event, marry):
        """Extract onset and duration information.

        Pull onset, duration values from self.df_events for
        indicies supplied with idx_event.

        Parameters
        ----------
        idx_event : list
            Indicies of self.df_events for behavior of interest
        marry : bool
            Whether to return AFNI-styled married onset:duration,
            or just onset times.

        Returns
        -------
        list
            Event onset or onset:duration times

        """
        onset = self.df_events.loc[idx_event, "onset"].tolist()
        duration = self.df_events.loc[idx_event, "duration"].tolist()
        if marry:
            return [f"{ons}:{dur}" for ons, dur in zip(onset, duration)]
        else:
            return [str(x) for x in onset]

    def common_events(self, marry=True, common_name=None):
        """Generate timing files for common events across both sessions.

        Make timing files for replay, judge, and wash events. Ouput timing
        file will use the 1D extension, use a BIDs-ish naming convention
        (sub-*_sess-*_task-*_desc-*_events.1D), and write the BIDS
        description field with "com[Rep|Jud|Was]" (e.g. desc-comRep).

        Fixations (fixS, fix) will be used as baseline in the deconvolution
        models, so timing files for these events are not generated.

        Parameters
        ----------
        marry : bool, optional
            Whether to generate timing file with AFNI-styled married
            onset:duration or just onset times.
        common_name : None or str, optional
            [replay | judge | wash]
            Generate a timing file for a specific event

        Returns
        -------
        list
            Paths to generated timing files

        Raises
        ------
        RunTimeError
            Generated timing file is empty
        TypeError
            Parameter for marry is not bool
        ValueError
            Parameter for common_name is not found in common_dict

        Notes
        -----
        Timing files written to self.subj_tf_dir

        """
        # Validate marry argument
        if not isinstance(marry, bool):
            raise TypeError("Argument 'marry' is bool")

        # Set basic trial types
        #   key = value in self.df_events["trial_types"]
        #   value = AFNI-style description
        common_dict = {
            "replay": "comRep",
            "judge": "comJud",
            "wash": "comWas",
        }

        # Validate user input and generate new common_dict
        if common_name:
            valid_list = [x for x in common_dict.keys()]
            if common_name not in valid_list:
                raise ValueError(
                    f"Inappropriate event supplied : {common_name}"
                )
            h_dict = {}
            h_dict[common_name] = common_dict[common_name]
            del common_dict
            common_dict = h_dict

        # Generate timing files for all events in common_dict
        out_list = []
        for event, tf_name in common_dict.items():

            # Make an empty file
            tf_path = os.path.join(
                self.subj_tf_dir,
                f"{self.subj}_{self.sess}_task-{self.task}_"
                + f"desc-{tf_name}_events.1D",
            )
            open(tf_path, "w").close()

            # Get event info for each run
            for run in self.events_run:

                # Identify index of events, make an AFNI line for events
                idx_event = self.df_events.index[
                    (self.df_events["trial_type"] == event)
                    & (self.df_events["run"] == run)
                ].tolist()
                ons_dur = self._onset_duration(idx_event, marry)
                line_content = " ".join(ons_dur)

                # Append line to timing file
                with open(tf_path, "a") as tf:
                    tf.writelines(f"{line_content}\n")

            # Check for content in timing file
            if os.stat(tf_path).st_size == 0:
                raise RuntimeError(f"Empty file detected : {tf_path}")
            else:
                out_list.append(tf_path)
        return out_list

    def select_events(self, marry=True, select_name=None):
        """Generate timing files for selection trials.

        Make timing files for trials where participant selects emotion
        or intensity from a list. Ouput timing file will use the 1D
        extension, use a BIDs-ish naming convention
        (sub-*_sess-*_task-*_desc-*_events.1D), and write the BIDS
        description field with "sel[Int|Emo]" (e.g. desc-selInt).

        This method could be accomplished by common_events method as the
        only real differences are found in select_dict. This method
        was written separately, however, in anticipation of future desires
        to compare the selection responses to the movie|scenario
        previously presented.

        Parameters
        ----------
        marry : bool, optional
            Whether to generate timing file with AFNI-styled married
            onset:duration or just onset times.
        select_name : None or str, optional
            [emotion | intensity]
            Generate a timing file for a specific event

        Returns
        -------
        list
            Paths to generated timing files

        Raises
        ------
        RunTimeError
            Generated timing file is empty
        TypeError
            Parameter for marry is not bool
        ValueError
            Parameter for select_name is not found in select_dict

        Notes
        -----
        Timing files written to self.subj_tf_dir

        """
        # Validate marry argument
        if not isinstance(marry, bool):
            raise TypeError("Argument 'marry' is bool")

        # Set selection trial types
        #   key = value in self.df_events["trial_types"]
        #   value = AFNI-style description
        select_dict = {
            "emotion": "selEmo",
            "intensity": "selInt",
        }

        # Validate user input and generate new select_dict
        if select_name:
            valid_list = [x for x in select_dict.keys()]
            if select_name not in valid_list:
                raise ValueError(
                    f"Inappropriate event supplied : {select_name}"
                )
            h_dict = {}
            h_dict[select_name] = select_dict[select_name]
            del select_dict
            select_dict = h_dict

        # Generate timing files for all events in select_dict
        out_list = []
        for select, tf_name in select_dict.items():

            # Make an empty file
            tf_path = os.path.join(
                self.subj_tf_dir,
                f"{self.subj}_{self.sess}_task-{self.task}_"
                + f"desc-{tf_name}_events.1D",
            )
            open(tf_path, "w").close()

            # Get event info for each run
            for run in self.events_run:

                # Identify index of events, make an AFNI line for events
                idx_select = self.df_events.index[
                    (self.df_events["trial_type"] == select)
                    & (self.df_events["run"] == run)
                ]
                ons_dur = self._onset_duration(idx_select, marry)
                line_content = " ".join(ons_dur)

                # Append line to timing file
                with open(tf_path, "a") as tf:
                    tf.writelines(f"{line_content}\n")

            # Check for content in timing file
            if os.stat(tf_path).st_size == 0:
                raise RuntimeError(f"Empty file detected : {tf_path}")
            else:
                out_list.append(tf_path)
        return out_list

    def session_events(self, marry=True, emotion_name=None, emo_query=False):
        """Generate timing files for session-specific stimulus trials.

        Make timing files for emotions presented during movies or scenarios.
        Ouput timing file will use the 1D extension, use a BIDs-ish naming
        convention (sub-*_sess-*_task-*_desc-*_events.1D), and write the BIDS
        description field with in a [mov|sce]Emotion format, using 3
        characters to identify the emotion. For example, movFea = movie with
        fear stimuli, sceCal = scenario with calmness stimuli.

        Parameters
        ----------
        marry : bool, optional
            Whether to generate timing file with AFNI-styled married
            onset:duration or just onset times.
        emotion_name : None or str, optional
            Generate a timing file for a specific emotion, see
            emo_query.
        emo_query : bool, optional
            Print a list of emotions, useful for specifying
            emotion_name.

        Returns
        -------
        list
            Paths to generated timing files, or emotions that will be
            extracted from events files (if emo_query=True).

        Notes
        -----
        Timing files written to self.subj_tf_dir

        """
        # Validate bool args
        if not isinstance(emo_query, bool):
            raise TypeError("Argument 'emo_query' is bool")
        if not isinstance(marry, bool):
            raise TypeError("Argument 'marry' is bool")

        # Set selection trial types
        #   key = value in self.df_events["emotion"]
        #   value = AFNI-style description
        sess_dict = {
            "amusement": "Amu",
            "anger": "Ang",
            "anxiety": "Anx",
            "awe": "Awe",
            "calmness": "Cal",
            "craving": "Cra",
            "disgust": "Dis",
            "excitement": "Exc",
            "fear": "Fea",
            "horror": "Hor",
            "joy": "Joy",
            "neutral": "Neu",
            "romance": "Rom",
            "sadness": "Sad",
            "surprise": "Sur",
        }

        # Provide emotions in sess_dict
        if emo_query:
            return [x for x in sess_dict.keys()]

        # Validate user input, generate emo_list
        if emotion_name:
            valid_list = [x for x in sess_dict.keys()]
            if emotion_name not in valid_list:
                raise ValueError(
                    f"Inappropriate emotion supplied : {emotion_name}"
                )
            emo_list = [emotion_name]
        else:

            # Identify unique emotions in dataframe
            trial_type_value = self.task[:-1]
            idx_sess = self.df_events.index[
                self.df_events["trial_type"] == trial_type_value
            ].tolist()
            emo_all = self.df_events.loc[idx_sess, "emotion"].tolist()
            emo_list = np.unique(np.array(emo_all)).tolist()

        # Generate timing files for all events in emo_list
        out_list = []
        for emo in emo_list:

            # Check that emo is found in planned dictionary
            if emo not in sess_dict.keys():
                raise ValueError(f"Unexpected emotion encountered : {emo}")

            # Determine timing file name, make an empty file
            tf_name = (
                f"mov{sess_dict[emo]}"
                if self.task == "movies"
                else f"sce{sess_dict[emo]}"
            )
            tf_path = os.path.join(
                self.subj_tf_dir,
                f"{self.subj}_{self.sess}_task-{self.task}_"
                + f"desc-{tf_name}_events.1D",
            )
            open(tf_path, "w").close()

            # Get emo info for each run
            for run in self.events_run:

                # Identify index of emotions, account for emotion
                # not occurring in current run, make appropriate
                # AFNI line for event.
                idx_emo = self.df_events.index[
                    (self.df_events["emotion"] == emo)
                    & (self.df_events["run"] == run)
                ].tolist()
                if not idx_emo:
                    line_content = "*"
                else:
                    ons_dur = self._onset_duration(idx_emo, marry)
                    line_content = " ".join(ons_dur)

                # Append line to timing file
                with open(tf_path, "a") as tf:
                    tf.writelines(f"{line_content}\n")

            # Check for content in timing file
            if os.stat(tf_path).st_size == 0:
                raise RuntimeError(f"Empty file detected : {tf_path}")
            else:
                out_list.append(tf_path)
        return out_list
