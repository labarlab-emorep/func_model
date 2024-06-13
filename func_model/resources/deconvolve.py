"""Resources for preparing, writing, and running AFNI-style GLMs.

TimingFiles : Make AFNI-style event timing files
MotionCensor : Make AFNI-style motion files
RunReml : Generated and execute deconvolution
ProjectRest : Generate correlation matrices

"""

import os
import json
import time
import math
import subprocess
import pandas as pd
import numpy as np
from func_model.resources import helper
from func_model.resources import submit


class TimingFiles:
    """Create timing files for various types of EmoRep events.

    Aggregate all BIDS task events files for a participant's session,
    and then generate AFNI-style timing files (row number == run number)
    with onset or onset:duration information for each line.

    Timing files are written to:
        <subj_work>/timing_files

    Parameters
    ----------
    subj : str
        BIDs subject identifier
    sess : str
        BIDs session identifier
    task : str
        BIDS task identifier
    subj_work : str, os.PathLike
        Location of working directory for intermediates
    sess_events : list
        Paths to subject, session BIDS events files sorted
        by run number

    Methods
    -------
    common_events()
        Generate and return timing files for common events
        (replay, judge, and wash)
    select_events()
        Generate and return timing files for selection trials
        (emotion, intensity)
    session_events()
        Generate and return timing files for movie or scenario emotions
    session_blocks()
        Generate and return timing files for movie/scenario emotion blocks

    Notes
    -----
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

        """
        print("\nInitializing TimingFiles")

        # Check arguments
        if len(sess_events) < 1:
            raise ValueError("Cannot make timing files from 0 events.tsv")
        if not helper.valid_task(task):
            raise ValueError(f"Expected task name : {task}")

        # Set attributes, make output location
        self._subj = subj
        self._sess = sess
        self._task = task
        self._sess_events = sess_events
        self._subj_tf_dir = os.path.join(subj_work, "timing_files")
        self._emo_switch = helper.emo_switch()
        if not os.path.exists(self._subj_tf_dir):
            os.makedirs(self._subj_tf_dir)

        # Generate dataframe from events files
        self._event_dataframe()

    def _event_dataframe(self):
        """Combine data from events files into dataframe."""
        # Read-in events files, construct list of dataframes. Determine
        # run info from file name.
        events_data = [pd.read_table(x) for x in self._sess_events]
        self._events_run = [
            int(x.split("_run-")[1].split("_")[0]) for x in self._sess_events
        ]
        if len(events_data) != len(self._events_run):
            raise ValueError("Number of runs and events files differ")

        # Add run info to listed dataframes, construct session dataframe
        for idx, _ in enumerate(events_data):
            events_data[idx]["run"] = self._events_run[idx]
        self._df_events = pd.concat(events_data).reset_index(drop=True)

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
            Derived from self._events_run
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

        Pull onset, duration values from self._df_events for
        indicies supplied with idx_event.

        Parameters
        ----------
        idx_event : list
            Indicies of self._df_events for behavior of interest
        marry : bool
            Whether to return AFNI-styled married onset:duration,
            or just onset times.

        Returns
        -------
        list
            Event onset or onset:duration times

        """
        onset = self._df_events.loc[idx_event, "onset"].tolist()
        duration = self._df_events.loc[idx_event, "duration"].tolist()
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
            {"replay", "judge", "wash"}
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
        Timing files written to self._subj_tf_dir

        """
        print("\tMaking timing files for common events")

        # Validate marry argument
        if not isinstance(marry, bool):
            raise TypeError("Argument 'marry' is bool")

        # Set basic trial types
        #   key = value in self._df_events["trial_types"]
        #   value = AFNI-style description
        common_dict = {
            "replay": "comRep",
            "judge": "comJud",
            "wash": "comWas",
        }

        # Validate user input and generate new common_dict
        if common_name:
            if common_name not in list(common_dict.keys()):
                raise ValueError(
                    f"Inappropriate event supplied : {common_name}"
                )
            common_dict = {common_name: common_dict[common_name]}

        # Generate timing files for all events in common_dict
        out_list = []
        for event, tf_name in common_dict.items():
            # Make an empty file
            tf_path = os.path.join(
                self._subj_tf_dir,
                f"{self._subj}_{self._sess}_{self._task}_"
                + f"desc-{tf_name}_events.1D",
            )
            open(tf_path, "w").close()

            # Get event info for each run
            for run in self._events_run:
                # Identify index of events, make an AFNI line for events
                idx_event = self._df_events.index[
                    (self._df_events["trial_type"] == event)
                    & (self._df_events["run"] == run)
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
            {"emotion", "intensity"}
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
        Timing files written to self._subj_tf_dir

        """
        print("\tMaking timing files for selection trials")

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
            if select_name not in list(select_dict.keys()):
                raise ValueError(
                    f"Inappropriate event supplied : {select_name}"
                )
            select_dict = {select_name: select_dict[select_name]}

        # Generate timing files for all events in select_dict
        out_list = []
        for select, tf_name in select_dict.items():
            # Make an empty file
            tf_path = os.path.join(
                self._subj_tf_dir,
                f"{self._subj}_{self._sess}_{self._task}_"
                + f"desc-{tf_name}_events.1D",
            )
            open(tf_path, "w").close()

            # Get event info for each run
            for run in self._events_run:
                # Identify index of events, make an AFNI line for events
                idx_select = self._df_events.index[
                    (self._df_events["trial_type"] == select)
                    & (self._df_events["run"] == run)
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

    def session_events(
        self,
        marry=True,
        emotion_name=None,
        emo_query=False,
    ):
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
        Timing files written to self._subj_tf_dir

        """
        print("\tMaking timing files for session-specific trials")

        # Validate bool args
        if not isinstance(emo_query, bool):
            raise TypeError("Argument 'emo_query' is bool")
        if not isinstance(marry, bool):
            raise TypeError("Argument 'marry' is bool")

        # Provide emotions in sess_dict
        if emo_query:
            return [x for x in self._emo_switch.keys()]

        # Validate user input, generate emo_list
        task_id = self._task.split("-")[1]
        if emotion_name:
            if emotion_name not in list(self._emo_switch.keys()):
                raise ValueError(
                    f"Inappropriate emotion supplied : {emotion_name}"
                )
            emo_list = [emotion_name]
        else:
            # Identify unique emotions in dataframe
            trial_type_value = task_id[:-1]
            idx_sess = self._df_events.index[
                self._df_events["trial_type"] == trial_type_value
            ].tolist()
            emo_all = self._df_events.loc[idx_sess, "emotion"].tolist()
            emo_list = np.unique(np.array(emo_all)).tolist()

        # Generate timing files for all events in emo_list
        out_list = []
        for emo in emo_list:
            # Check that emo is found in planned dictionary
            if emo not in self._emo_switch.keys():
                raise ValueError(f"Unexpected emotion encountered : {emo}")

            # Determine timing file name, make an empty file
            tf_name = task_id[:3] + self._emo_switch[emo]
            tf_path = os.path.join(
                self._subj_tf_dir,
                f"{self._subj}_{self._sess}_{self._task}_"
                + f"desc-{tf_name}_events.1D",
            )
            open(tf_path, "w").close()

            # Get emo info for each run
            for run in self._events_run:

                # Identify index of emotions, account for emotion
                # not occurring in current run, make appropriate
                # AFNI line for event.
                idx_emo = self._df_events.index[
                    (self._df_events["emotion"] == emo)
                    & (self._df_events["run"] == run)
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

    def session_blocks(self):
        """Generate timing files for session-specific stimulus blocks.

        Ouput timing file will use the 1D extension, a BIDs-ish naming
        convention (sub-*_sess-*_task-*_desc-*_events.1D), write the BIDS
        description field with a blk[Mov|Sce]Emotion format, using 3
        characters to identify the emotion. For example, blkMovFea = movie
        block with fear stimuli, blkSceCal = scenario with calmness stimuli.

        Returns
        -------
        list
            Paths to generated timing files

        Notes
        -----
        Timing files written to self._subj_tf_dir

        """
        # Identify unique emotions in dataframe
        task_id = self._task.split("-")[1]
        trial_type_value = task_id[:-1]
        idx_sess = self._df_events.index[
            self._df_events["trial_type"] == trial_type_value
        ].tolist()
        emo_all = self._df_events.loc[idx_sess, "emotion"].tolist()
        emo_list = np.unique(np.array(emo_all)).tolist()

        # Generate timing files for all events in emo_list
        out_list = []
        for emo in emo_list:

            # Check that emo is found in planned dictionary
            if emo not in self._emo_switch.keys():
                raise ValueError(f"Unexpected emotion encountered : {emo}")

            # Determine timing file name, make an empty file
            tf_name = "blk" + task_id.capitalize()[0] + self._emo_switch[emo]
            tf_path = os.path.join(
                self._subj_tf_dir,
                f"{self._subj}_{self._sess}_{self._task}_"
                + f"desc-{tf_name}_events.1D",
            )
            open(tf_path, "w").close()

            # Get emo info for each run
            # dur_list = []
            for run in self._events_run:

                # Identify index of emotions, account for emotion
                # not occurring in current run, make appropriate
                # AFNI line for event.
                idx_emo = self._df_events.index[
                    (self._df_events["emotion"] == emo)
                    & (self._df_events["run"] == run)
                ].tolist()

                if not idx_emo:
                    line_content = "*"
                else:

                    # Find onsets and durations
                    onset = self._df_events.loc[idx_emo, "onset"].tolist()
                    duration = self._df_events.loc[
                        idx_emo, "duration"
                    ].tolist()

                    # Calculate block length, make married line
                    block_onset = onset[0]
                    block_end = onset[-1] + duration[-1]
                    block_dur = round(block_end - block_onset, 2)
                    line_content = f"{block_onset}:{block_dur}"

                # Append line to timing file
                with open(tf_path, "a") as tf:
                    tf.writelines(f"{line_content}\n")

            # Check for content in timing file
            if os.stat(tf_path).st_size == 0:
                raise ValueError(f"Empty file detected : {tf_path}")
            else:
                out_list.append(tf_path)
        return out_list


class MotionCensor:
    """Make motion and censor files for AFNI deconvolution.

    Mine fMRIPrep timeseries.tsv files for required information,
    and generate files in a format AFNI can use in 3dDeconvolve.

    Parameters
    ----------
    subj_work : path
        Location of working directory for intermediates
    proj_deriv : path
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories
    func_motion : list
        Locations of timeseries.tsv files produced by fMRIPrep,
        file names must follow BIDS convention

    Methods
    -------
    mean_motion()
        Make average motion file for 6 dof
    deriv_motion()
        Make derivative motion file for 6 dof
    censor_volumes()
        Determine which volumes to censor
    count_motion()
        Determine number, proportion of censored volumes

    Notes
    -----
    -   As runs do not have an equal number of volumes, motion/censor files
        for each run are concatenated into a single file rather than
        managing zero padding.
    -   Output formats use 1D extension since AFNI assumes TSVs
        contain a header.

    """

    def __init__(self, subj_work, proj_deriv, func_motion):
        """Setup for making motion, censor files.

        Set attributes, make output directory, setup basic
        output file name.

        Attributes
        ----------
        _out_dir : path
            Output directory for motion, censor files
        _out_str : str
            Basic output file name
        _sing_prep : list
            First part of subprocess call for AFNI singularity

        Raises
        ------
        TypeError
            Improper parameter types
        ValueError
            Improper naming convention of motion timeseries file

        """
        # Validate args and setup
        if not isinstance(func_motion, list):
            raise TypeError("func_motion requires list type")

        print("\nInitializing MotionCensor")
        try:
            subj, sess, task, _, desc, _ = os.path.basename(
                func_motion[0]
            ).split("_")
        except ValueError:
            raise ValueError(
                "BIDS file names required for items in func_motion: "
                + "subject, session, task, run, description, and suffix.ext "
                + "BIDS fields are required by afni.MotionCensor."
            )
        self._out_str = f"{subj}_{sess}_{task}_{desc}_timeseries.1D"
        self._proj_deriv = proj_deriv
        self._func_motion = func_motion
        self._subj_work = subj_work
        self._sing_prep = helper.prepend_afni_sing(
            self._proj_deriv, self._subj_work
        )
        self._out_dir = os.path.join(subj_work, "motion_files")
        if not os.path.exists(self._out_dir):
            os.makedirs(self._out_dir)

    def _write_df(self, df_out, name, col_names=None):
        """Write dataframe to output location.

        Special output formatting for AFNI compatibility.

        Parameters
        ----------
        df_out : pd.DataFrame
            A dataframe of motion or censor info
        name : str
            Identifier, will be written
            to BIDS description field
        col_names : list, optional
            Select columns to write out

        Returns
        -------
        path
            Location of output dataframe

        """
        out_path = os.path.join(
            self._out_dir, self._out_str.replace("confounds", name)
        )
        df_out.to_csv(
            out_path,
            sep="\t",
            index=False,
            header=False,
            float_format="%.6f",
            columns=col_names,
        )
        return out_path

    def mean_motion(self):
        """Make file for mean motion events.

        References the [trans|rot]_[x|y|z] columns
        to get all 6 dof.

        Attributes
        ----------
        mean_path : path
            Location of mean motion file

        Returns
        -------
        path
            Location of mean motion file

        """
        print("\tMaking mean motion file")

        # Set column identifiers
        labels_mean = [
            "trans_x",
            "trans_y",
            "trans_z",
            "rot_x",
            "rot_y",
            "rot_z",
        ]

        # Extract relevant columns for each run
        mean_cat = []
        for mot_path in self._func_motion:
            df = pd.read_csv(mot_path, sep="\t")
            df_mean = df[labels_mean]
            df_mean = df_mean.round(6)
            mean_cat.append(df_mean)

        # Combine runs, write out
        df_mean_all = pd.concat(mean_cat, ignore_index=True)
        mean_out = self._write_df(df_mean_all, "mean")
        self._mean_path = mean_out
        return mean_out

    def deriv_motion(self):
        """Make file for derivative motion events.

        References the [trans|rot]_[x|y|z]_deriv1 columns
        to get all 6 dof.

        Returns
        -------
        path
            Location of derivative motion file

        """
        print("\tMaking derivative motion file")

        # Set column identifiers
        labels_deriv = [
            "trans_x_derivative1",
            "trans_y_derivative1",
            "trans_z_derivative1",
            "rot_x_derivative1",
            "rot_y_derivative1",
            "rot_z_derivative1",
        ]

        # Extract relevant columns for each run
        deriv_cat = []
        for mot_path in self._func_motion:
            df = pd.read_csv(mot_path, sep="\t")
            df_drv = df[labels_deriv]
            df_drv = df_drv.fillna(0)
            df_drv = df_drv.round(6)
            deriv_cat.append(df_drv)

        # Combine runs, write out
        df_deriv_all = pd.concat(deriv_cat, ignore_index=True)
        deriv_out = self._write_df(df_deriv_all, "deriv")
        return deriv_out

    def censor_volumes(self, thresh=0.3):
        """Determine volumes needing censoring.

        Generate AFNI-styled censor files. Converts fMRIPrep rotation
        values (radians) into millimeters.

        Parameters
        ----------
        thresh : float
            Min/Max rotation value (mm)

        Attributes
        ----------
        df_censor : pd.DataFrame
            Row = volume
            Value = binary censor due to motion

        Returns
        -------
        path
            location of censor file

        Notes
        -----
        -   Requires self._mean_path, will trigger method.

        """
        # Check for required attribute, trigger
        if not hasattr(self, "_mean_path"):
            self.mean_motion()

        # Setup output path, avoid repeating work
        out_path = os.path.join(
            self._out_dir, self._out_str.replace("confounds", "censor")
        )
        if os.path.exists(out_path):
            self._df_censor = pd.read_csv(out_path, header=None)
            return out_path

        # Find significant motion events
        print("\tMaking censor file")
        bash_list = [
            "1d_tool.py",
            f"-infile {self._mean_path}",
            "-derivative",
            "-collapse_cols weighted_enorm",
            "-weight_vec 1 1 1 57.3 57.3 57.3",
            f"-moderate_mask -{thresh} {thresh}",
            f"-write_censor {out_path}",
        ]
        bash_cmd = " ".join(self._sing_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_path, "Censor")
        self._df_censor = pd.read_csv(out_path, header=None)
        return out_path

    def count_motion(self):
        """Quantify missing volumes due to motion.

        Calculate number and proportion of volumes that
        necessitate censoring. Save info to JSON file.

        Returns
        -------
        path
            Location of censored info file

        Notes
        -----
        -   Requires self._df_censor, will trigger method.
        -   Writes totals to self._out_dir/info_censored_volumes.json

        """
        print("\tCounting censored volumes")

        # Check for required attribute, trigger
        if not hasattr(self, "_df_censor"):
            self.censor_volumes()

        # Quick calculations
        num_vol = self._df_censor[0].sum()
        num_tot = len(self._df_censor)
        cen_dict = {
            "total_volumes": int(num_tot),
            "included_volumes": int(num_vol),
            "proportion_excluded": round(1 - (num_vol / num_tot), 3),
        }

        # Write out and return
        out_path = os.path.join(self._out_dir, "info_censored_volumes.json")
        with open(out_path, "w") as jfile:
            json.dump(cen_dict, jfile)
        return out_path


class _WriteDecon:
    """Write an AFNI 3dDeconvolve command.

    Write 3dDeconvolve command supporting different basis functions
    and data types (task, resting-state).

    Parameters
    ----------
    subj_work : str, os.PathLike
        Location of working directory for intermediates
    proj_deriv : str, os.PathLike
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories.
    sess_func : dict
        Contains reference names (key) and paths (value) to
        preprocessed functional files.
        Required keys:
        -   [func-scaled] = list of scaled EPI file paths
        -   [mot-mean] = path to mean motion regressor
        -   [mot-deriv] = path to derivative motion regressor
        -   [mot-cens] = path to censor vector
    sess_anat : dict
        Contains reference names (key) and paths (value) to
        preprocessed anatomical files.
        Required keys:
        -   [mask-int] = path to intersection mask
        -   [mask-min] = path to minimum value mask
        -   [mask-CSe] = path to eroded CSF mask

    Attributes
    ----------
    decon_cmd : str
        Generated 3dDeconvolve command
    decon_name : str
        Prefix for output deconvolve files
    epi_masked : str, os.PathLike
        Location of masked EPI data

    Methods
    -------
    build_decon()
        Trigger the appropriate method for the current pipeline, e.g.
        build_decon(model_name="univ") causes the method "write_univ"
        to be executed.
    write_univ()
        Write a univariate 3dDeconvolve command for sanity checks
    write_rest()
        Setup for and write a resting-state 3dDeconvolve command

    """

    def __init__(
        self,
        subj_work,
        proj_deriv,
        sess_func,
        sess_anat,
    ):
        """Initialize object."""
        # Validate dict keys
        for _key in ["func-scaled", "mot-mean", "mot-deriv", "mot-cens"]:
            if _key not in sess_func:
                raise KeyError(f"Expected {_key} key in sess_func")
        for _key in ["mask-int", "mask-min", "mask-CSe"]:
            if _key not in sess_anat:
                raise KeyError(f"Expected {_key} key in sess_anat")
        if len(sess_func["func-scaled"]) == 0:
            raise ValueError("Expected list of paths to scaled EPI files.")

        print("\nInitializing WriteDecon")
        self._proj_deriv = proj_deriv
        self._subj_work = subj_work
        self._sess_func = sess_func
        self._sess_anat = sess_anat
        self._afni_prep = helper.prepend_afni_sing(
            self._proj_deriv, self._subj_work
        )

    def build_decon(self, model_name, sess_tfs=None):
        """Trigger deconvolution method.

        Use model_name to trigger the method that writes the
        relevant 3dDeconvolve command.

        Parameters
        ----------
        model_name : str
            {"univ", "rest", "mixed"}
            Desired AFNI model, triggers corresponding methods
        sess_tfs : None, dict, optional
            Required by model_name = univ|indiv.
            Contains reference names (key) and paths (value) to
            session AFNI-style timing files.

        """
        # Validate model name
        model_valid = helper.valid_models(model_name)
        if not model_valid:
            raise ValueError(f"Unsupported model name : {model_name}")

        # Require timing files for task decons
        if model_name in ["univ", "mixed"]:
            if not sess_tfs:
                raise ValueError(
                    f"Argument sess_tfs required with model_name={model_name}"
                )
            self._tf_dict = sess_tfs

        # Find, trigger appropriate method
        write_meth = getattr(self, f"write_{model_name}")
        write_meth()

    def _build_behavior(self, basis_func):
        """Build a behavior regressor argument.

        Build a 3dDeconvolve behavior regressor accounting
        for desired basis function. Use with task deconvolutions
        (not resting-state pipelines).

        Parameters
        ----------
        basis_func : str
            {"dur_mod", "ind_mod"}
            Desired basis function for behaviors in 3dDeconvolve

        Returns
        -------
        str
            Behavior regressor for 3dDeconvolve

        """
        # Validate
        if basis_func not in ["dur_mod", "ind_mod"]:
            raise ValueError(f"Unsupported basis_func: {basis_func}")

        # Build regressor for each behavior
        print("\t\tBuilding behavior regressors ...")
        model_beh = []
        for tf_name, tf_path in self._tf_dict.items():
            self._count_beh += 1

            # Make stim line based on basis_func
            if basis_func == "dur_mod":
                model_beh.append(
                    f"-stim_times_AM1 {self._count_beh} {tf_path} 'dmBLOCK(1)'"
                )
            elif basis_func == "ind_mod":
                model_beh.append(
                    f"-stim_times_IM {self._count_beh} {tf_path} 'dmBLOCK(1)'"
                )
            model_beh.append(f"-stim_label {self._count_beh} {tf_name}")

        # Combine into string for 3dDeconvolve parameter
        return " ".join(model_beh)

    def write_univ(self, basis_func="dur_mod", decon_name="decon_univ"):
        """Write an AFNI 3dDeconvolve command for univariate checking.

        Build 3dDeconvolve command with minimal support for different
        basis functions.

        Parameters
        ----------
        basis_func : str, optional
            {"dur_mod", "ind_mod"}
            Desired basis function for behaviors in 3dDeconvolve
        decon_name : str, optional
            Prefix for output deconvolve files

        Attributes
        ----------
        decon_cmd : str
            Generated 3dDeconvolve command
        decon_name : str
            Prefix for output deconvolve files

        """
        # Validate
        if basis_func not in ["dur_mod", "ind_mod"]:
            raise ValueError("Invalid basis_func parameter")

        # Determine input variables for 3dDeconvolve
        print("\tBuilding 3dDeconvolve command ...")
        epi_preproc = " ".join(self._sess_func["func-scaled"])
        reg_motion_mean = self._sess_func["mot-mean"]
        reg_motion_deriv = self._sess_func["mot-deriv"]
        motion_censor = self._sess_func["mot-cens"]
        mask_int = self._sess_anat["mask-int"]

        # Build behavior regressors
        self._count_beh = 0
        reg_events = self._build_behavior(basis_func)

        # Build decon command
        decon_list = [
            "3dDeconvolve",
            "-x1D_stop",
            "-GOFORIT",
            f"-mask {mask_int}",
            f"-input {epi_preproc}",
            f"-censor {motion_censor}",
            f"-ortvec {reg_motion_mean} mot_mean",
            f"-ortvec {reg_motion_deriv} mot_deriv",
            "-polort A",
            "-float",
            "-local_times",
            f"-num_stimts {self._count_beh}",
            reg_events,
            "-jobs 1",
            f"-x1D {self._subj_work}/X.{decon_name}.xmat.1D",
            f"-xjpeg {self._subj_work}/X.{decon_name}.jpg",
            f"-x1D_uncensored {self._subj_work}/X.{decon_name}.jpg",
            f"-bucket {self._subj_work}/{decon_name}_stats",
            f"-cbucket {self._subj_work}/{decon_name}_cbucket",
            f"-errts {self._subj_work}/{decon_name}_errts",
        ]
        self.decon_cmd = " ".join(self._afni_prep + decon_list)

        # Write script for review, records
        decon_script = os.path.join(self._subj_work, f"{decon_name}.sh")
        with open(decon_script, "w") as script:
            script.write(self.decon_cmd)
        self.decon_name = decon_name

    def write_indiv(self):
        """Write an AFNI 3dDeconvolve command for individual mod checking.

        DEPRECATED.

        The "indiv" approach requires the same files and workflow as "univ",
        the only difference is in the basis function (and file name). So,
        use the class method write_univ with different parameters.

        """
        self.write_univ(basis_func="ind_mod", decon_name="decon_indiv")

    def write_mixed(self, decon_name="decon_mixed"):
        """Write an AFNI 3dDeconvolve command for mixed modelling.

        Regressors include event and block designs.

        """
        # Determine input variables for 3dDeconvolve
        print("\tBuilding 3dDeconvolve command ...")
        epi_preproc = " ".join(self._sess_func["func-scaled"])
        reg_motion_mean = self._sess_func["mot-mean"]
        reg_motion_deriv = self._sess_func["mot-deriv"]
        motion_censor = self._sess_func["mot-cens"]
        mask_int = self._sess_anat["mask-int"]

        # Build behavior regressors
        self._count_beh = 0
        reg_events = self._build_behavior("dur_mod")

        # Build decon command
        decon_list = [
            "3dDeconvolve",
            "-x1D_stop",
            "-GOFORIT",
            f"-mask {mask_int}",
            f"-input {epi_preproc}",
            f"-censor {motion_censor}",
            f"-ortvec {reg_motion_mean} mot_mean",
            f"-ortvec {reg_motion_deriv} mot_deriv",
            "-polort A",
            "-float",
            "-local_times",
            f"-num_stimts {self._count_beh}",
            reg_events,
            "-jobs 1",
            f"-x1D {self._subj_work}/X.{decon_name}.xmat.1D",
            f"-xjpeg {self._subj_work}/X.{decon_name}.jpg",
            f"-x1D_uncensored {self._subj_work}/X.{decon_name}.jpg",
            f"-bucket {self._subj_work}/{decon_name}_stats",
            f"-cbucket {self._subj_work}/{decon_name}_cbucket",
            f"-errts {self._subj_work}/{decon_name}_errts",
        ]
        self.decon_cmd = " ".join(self._afni_prep + decon_list)

        # Write script for review, records
        decon_script = os.path.join(self._subj_work, f"{decon_name}.sh")
        with open(decon_script, "w") as script:
            script.write(self.decon_cmd)
        self.decon_name = decon_name

    def write_rest(self, decon_name="decon_rest"):
        """Write an AFNI 3dDeconvolve command for resting state checking.

        First conduct PCA on the CSF to determine a nuissance timeseries.
        Then model all nuissance parameters to produce 'cleaned' resting
        state output. Writes a shell script to:
            <subj_work>/<decon_name>.sh

        Parameters
        ----------
        decon_name : str, optional
            Prefix for output deconvolve files

        Attributes
        ----------
        decon_cmd : str
            Generated 3dDeconvolve command
        decon_name : str
            Prefix for output deconvolve files

        """
        # Conduct principal components analysis
        pcomp_path = self._run_pca()

        # Unpack dictionaries for readability
        epi_path = self._sess_func["func-scaled"][0]
        censor_path = self._sess_func["mot-cens"]
        reg_motion_mean = self._sess_func["mot-mean"]
        reg_motion_deriv = self._sess_func["mot-deriv"]

        # Build deconvolve command, write script for review.
        # This will load effects of no interest on fitts sub-brick, and
        # errts will contain cleaned time series.
        print("\tBuilding 3dDeconvolve command ...")
        decon_list = [
            "3dDeconvolve",
            "-x1D_stop",
            f"-input {epi_path}",
            f"-censor {censor_path}",
            f"-ortvec {pcomp_path} csf_ts",
            f"-ortvec {reg_motion_mean} mot_mean",
            f"-ortvec {reg_motion_deriv} mot_deriv",
            "-polort A",
            "-fout -tout",
            f"-x1D {self._subj_work}/X.{decon_name}.xmat.1D",
            f"-xjpeg {self._subj_work}/X.{decon_name}.jpg",
            "-x1D_uncensored",
            f"{self._subj_work}/X.{decon_name}.nocensor.xmat.1D",
            f"-fitts {self._subj_work}/{decon_name}_fitts",
            f"-errts {self._subj_work}/{decon_name}_errts",
            f"-bucket {self._subj_work}/{decon_name}_stats",
        ]
        decon_cmd = " ".join(self._afni_prep + decon_list)

        # Write script for review/records
        decon_script = os.path.join(self._subj_work, f"{decon_name}.sh")
        with open(decon_script, "w") as script:
            script.write(decon_cmd)

        self.decon_cmd = decon_cmd
        self.decon_name = decon_name

    def _run_pca(self):
        """Conduct principal components analysis.

        Determine first 3 components of CSF 'signal'.

        Attributes
        ----------
        epi_masked : path
            Location of masked EPI data

        Returns
        -------
        path
            Location of file containing PCA output for each volume

        """
        # Setup output paths/names, avoid repeating work
        mask_name = "tmp_masked_" + os.path.basename(
            self._sess_func["func-scaled"][0]
        )
        epi_masked = os.path.join(self._subj_work, mask_name)
        out_name = os.path.basename(self._sess_func["mot-cens"]).replace(
            "censor", "csfPC"
        )
        out_path = os.path.join(self._subj_work, out_name)
        if os.path.exists(epi_masked) and os.path.exists(out_path):
            self.epi_masked = epi_masked
            return out_path

        # Remove bad volumes
        print("\t\tConducting CSF PCA")
        bash_list = [
            "3dcalc",
            f"-a {self._sess_func['func-scaled'][0]}",
            f"-b {self._sess_anat['mask-min']}",
            "-expr 'a*b'",
            f"-prefix {epi_masked}",
        ]
        bash_cmd = " ".join(self._afni_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, epi_masked, "Mask rest")

        # Get protocol info, calculate polynomial order
        epi_info = self._get_epi_info()
        num_pol = 1 + math.ceil(
            (epi_info["sum_vol"] * epi_info["len_tr"]) / 150
        )

        # Split censor file into runs
        out_cens = os.path.join(
            self._subj_work,
            f"tmp_{os.path.basename(self._sess_func['mot-cens'])}",
        )
        bash_list = [
            "1d_tool.py",
            f"-set_run_lengths {epi_info['sum_vol']}",
            "-select_runs 1",
            f"-infile {self._sess_func['mot-cens']}",
            f"-write {out_cens}",
        ]
        bash_cmd = " ".join(self._afni_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_cens, "Cens rest")

        # Censor EPI data
        out_proj = os.path.join(
            self._subj_work,
            f"tmp_proj_{os.path.basename(self._sess_func['func-scaled'][0])}",
        )
        bash_list = [
            "3dTproject",
            f"-polort {num_pol}",
            f"-prefix {out_proj}",
            f"-censor {out_cens}",
            "-cenmode KILL",
            f"-input {epi_masked}",
        ]
        bash_cmd = " ".join(self._afni_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_proj, "Proj rest")

        # Conduct PCA
        out_pcomp = os.path.join(self._subj_work, "tmp_pcomp")
        bash_list = [
            "3dpc",
            f"-mask {self._sess_anat['mask-CSe']}",
            "-pcsave 3",
            f"-prefix {out_pcomp}",
            out_proj,
        ]
        bash_cmd = " ".join(self._afni_prep + bash_list)
        _ = submit.submit_subprocess(
            bash_cmd, f"{out_pcomp}_vec.1D", "Pcomp rest"
        )

        # Account for censoring in PCA output
        tmp_out = os.path.join(self._subj_work, "test_" + out_name)
        bash_list = [
            "1d_tool.py",
            f"-censor_fill_parent {out_cens}",
            f"-infile {out_pcomp}_vec.1D",
            f"-write {tmp_out}",
        ]
        bash_cmd = " ".join(self._afni_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, tmp_out, "Split rest1")

        # Finalize PCA file
        bash_list = [
            "1d_tool.py",
            f"-set_run_lengths {epi_info['sum_vol']}",
            "-pad_into_many_runs 1 1",
            f"-infile {tmp_out}",
            f"-write {out_path}",
        ]
        bash_cmd = " ".join(self._afni_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_path, "Split rest")
        self.epi_masked = epi_masked
        return out_path

    def _get_epi_info(self):
        """Return dict of TR, volume info.

        Returns
        -------
        dict
            len_tr = float, TR value
            run_len = list, time (seconds) of each run
            run_vol = list, number of volumes in each run
            sum_vol = float, int, total number of volumes in session

        """
        # Find TR length
        bash_cmd = f"""
            fslhd \
                {self._sess_func["func-scaled"][0]} | \
                grep 'pixdim4' | \
                awk '{{print $2}}'
        """
        h_sp = subprocess.Popen(bash_cmd, shell=True, stdout=subprocess.PIPE)
        h_out, h_err = h_sp.communicate()
        h_sp.wait()
        len_tr = float(h_out.decode("utf-8").strip())

        # Get number of volumes and length (seconds) of each run
        run_len = []
        num_vol = []
        for epi_file in self._sess_func["func-scaled"]:
            # Extract number of volumes
            bash_cmd = f"""
                fslhd \
                    {self._sess_func["func-scaled"][0]} | \
                    grep dim4 | \
                    head -n 1 | \
                    awk '{{print $2}}'
            """
            h_sp = subprocess.Popen(
                bash_cmd, shell=True, stdout=subprocess.PIPE
            )
            h_out, h_err = h_sp.communicate()
            h_sp.wait()

            # Interpret, get number of volumes and run length
            h_vol = int(h_out.decode("utf-8").strip())
            run_len.append(str(h_vol * len_tr))
            num_vol.append(h_vol)

        # Find total number of volumes
        sum_vol = sum(num_vol)
        return {
            "len_tr": len_tr,
            "run_len": run_len,
            "run_vol": num_vol,
            "sum_vol": sum_vol,
        }


class RunReml(_WriteDecon):
    """Run 3dREMLfit deconvolution.

    Inherits _WriteDecon.

    Setup for and execute 3dREMLfit command generated by
    3dDeconvolve (afni.WriteDecon.build_decon).

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    subj_work : str, os.PathLike
        Location of working directory for intermediates
    proj_deriv : str, os.PathLike
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories.
    sess_anat : dict
        Contains reference names (key) and paths (value) to
        preprocessed anatomical files.
        Required keys:
        -   [mask-WMe] = path to eroded WM mask
    sess_func : dict
        Contains reference names (key) and paths (value) to
        preprocessed functional files.
        Required keys:
        -   [func-scaled] = list of scaled EPI paths
    log_dir : str, os.PathLike
        Output location for log files and scripts

    Methods
    -------
    generate_reml(subj, sess, decon_cmd, decon_name)
        Execute 3dDeconvolve to generate 3dREMLfit command
    exec_reml(subj, sess, reml_path, decon_name)
        Setup for and run 3dREMLfit

    Example
    -------
    run_reml = deconvolve.RunReml(*args)
    run_reml.build_decon(*args)
    run_reml.generate_reml()
    run_reml.exec_reml()

    """

    def __init__(
        self,
        subj,
        sess,
        subj_work,
        proj_deriv,
        sess_anat,
        sess_func,
        log_dir,
    ):
        """Initialize object."""
        # Validate needed keys
        if "mask-WMe" not in sess_anat:
            raise KeyError("Expected mask-WMe key in sess_anat")
        if "func-scaled" not in sess_func:
            raise KeyError("Expected func-scaled key in sess_func")

        print("\nInitializing RunDecon")
        self._subj = subj
        self._sess = sess
        self._subj_work = subj_work
        self._sess_anat = sess_anat
        self._sess_func = sess_func
        self._log_dir = log_dir
        self._afni_prep = helper.prepend_afni_sing(proj_deriv, subj_work)
        super().__init__(subj_work, proj_deriv, sess_func, sess_anat)

    def generate_reml(self):
        """Generate matrices and 3dREMLfit command.

        Run the 3dDeconvolve command to generate the 3dREMLfit
        command and required input.

        Returns
        -------
        path
            Location of 3dREMLfit script

        """
        # Setup output file, avoid repeating work
        print("\tRunning 3dDeconvolve command ...")
        self._reml_path = os.path.join(
            self._subj_work, f"{self.decon_name}_stats.REML_cmd"
        )
        if os.path.exists(self._reml_path):
            return

        # Execute decon_cmd, wait for singularity to close
        _, _ = submit.submit_sbatch(
            self.decon_cmd,
            f"dcn{self._subj[-4:]}s{self._sess[-1]}",
            self._log_dir,
            mem_gig=10,
        )
        if not os.path.exists(self._reml_path):
            time.sleep(30)

        # Check generated file length, account for 0 vs 1 indexing
        with open(self._reml_path, "r") as rf:
            for line_count, _ in enumerate(rf):
                pass
        line_count += 1
        if line_count != 8:
            raise ValueError(
                f"Expected 8 lines in {self._reml_path}, found {line_count}"
            )

    def _make_nuiss(self):
        """Make noise estimation file.

        Generate file containing white matter EPI signal, used with
        -dsort option in 3dREMLfit.

        Returns
        -------
        path
            Location of WM signal file

        """
        print("\tMaking nuissance mask")

        # Setup output location and name, avoid repeating work
        h_name = os.path.basename(self._sess_anat["mask-WMe"])
        out_path = os.path.join(
            self._subj_work, h_name.replace("label-WMe", "desc-nuiss")
        )
        if os.path.exists(out_path):
            return out_path

        # Concatenate EPI runs
        out_tcat = os.path.join(self._subj_work, "tmp_tcat_all-runs.nii.gz")
        bash_list = [
            "3dTcat",
            f"-prefix {out_tcat}",
            " ".join(self._sess_func["func-scaled"]),
        ]
        bash_cmd = " ".join(self._afni_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_tcat, "Tcat runs")

        # Make eroded mask in EPI time
        out_erode = os.path.join(self._subj_work, "tmp_eroded_all-runs.nii.gz")
        bash_list = [
            "3dcalc",
            f"-a {out_tcat}",
            f"-b {self._sess_anat['mask-WMe']}",
            "-expr 'a*bool(b)'",
            "-datum float",
            f"-prefix {out_erode}",
        ]
        bash_cmd = " ".join(self._afni_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_erode, "Erode")

        # Generate blurred WM file
        bash_list = [
            "3dmerge",
            "-1blur_fwhm 20",
            "-doall",
            f"-prefix {out_path}",
            out_erode,
        ]
        bash_cmd = " ".join(self._afni_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_path, "Nuiss")
        return out_path

    def exec_reml(self):
        """Exectue 3dREMLfit command.

        Setup for and exectue 3dREMLfit command generated
        by 3dDeconvolve. Writes reml command to:
            <subj_work>/decon_reml.sh

        Returns
        -------
        path
            Location of deconvolved HEAD file

        """
        # Set final path name (anticipate AFNI output)
        out_path = self._reml_path.replace(".REML_cmd", "_REML+tlrc.HEAD")
        if os.path.exists(out_path):
            return out_path

        # Extract reml command from generated reml_path
        tail_path = os.path.join(self._subj_work, "decon_reml.txt")
        bash_cmd = f"tail -n 6 {self._reml_path} > {tail_path}"
        _ = submit.submit_subprocess(bash_cmd, tail_path, "Tail")

        # Split reml command into lines, remove formatting
        with open(tail_path, "r", encoding="UTF-8") as tf:
            line_list = [line.rstrip() for line in tf]
        line_list = [x.replace("\\", "") for x in line_list]

        # Convert reml command into list
        reml_list = []
        for count, content in enumerate(line_list):
            # Keep -input param together but deal with double
            # quotes for compatibilty with sbatch --wrap
            if count == 1:
                reml_list.append(content.strip().replace('"', "'"))
            else:
                for word in content.split():
                    reml_list.append(word)

        # Remove trailing tcsh syntax
        idx_verb = reml_list.index("-verb")
        reml_list = reml_list[: idx_verb + 1]

        # Check that converting from file to list worked
        if reml_list[0] != "3dREMLfit":
            raise ValueError(
                "An issue occurred when converting "
                + f"contents of {self._reml_path} to a list"
            )

        # Make nuissance file, add to reml command
        nuiss_path = self._make_nuiss()
        reml_list.append(f"-dsort {nuiss_path}")
        reml_list.append("-GOFORIT")

        # Write script for review/records, then run
        print("\tRunning 3dREMLfit")
        bash_cmd = " ".join(self._afni_prep + reml_list)
        reml_script = os.path.join(
            self._subj_work, f"{self.decon_name}_reml.sh"
        )
        with open(reml_script, "w") as script:
            script.write(bash_cmd)

        wall_time = 18 if self.decon_name == "decon_rest" else 38
        _, _ = submit.submit_sbatch(
            bash_cmd,
            f"rml{self._subj[6:]}s{self._sess[-1]}",
            self._log_dir,
            num_hours=wall_time,
            num_cpus=6,
            mem_gig=24,
        )
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Expected to find {out_path}")
        return out_path


class ProjectRest:
    """Project a correlation matrix for resting state fMRI data.

    Execute generated 3ddeconvolve to produce a no-censor x-matrix,
    project correlation matrix accounting for WM and CSF nuissance,
    and conduct a seed-based correlation analysis.

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    subj_work : path
        Location of working directory for intermediates
    log_dir : path
        Output location for log files and scripts

    Methods
    -------
    gen_matrix(decon_cmd, decon_name)
        Execute 3ddeconvolve to make no-censor matrix
    anaticor(decon_name, epi_masked, anat_dict, func_dict)
        Generate regression matrix using anaticor method
    seed_corr(anat_dict)
        Generate seed-based correlation matrix

    """

    def __init__(self, subj, sess, subj_work, proj_deriv, log_dir):
        """Initialize object."""
        print("\nInitializing ProjectRest")
        self._subj = subj
        self._sess = sess
        self._subj_work = subj_work
        self._log_dir = log_dir
        self._afni_prep = helper.prepend_afni_sing(proj_deriv, self._subj_work)

    def gen_xmatrix(self, decon_cmd, decon_name):
        """Execute generated 3dDeconvolve command to make x-files.

        Cue 90's theme.

        Parameters
        ----------
        decon_cmd : str, afni.WriteDecon.build_decon.decon_cmd
            Bash 3dDeconvolve command
        decon_name : str, afni.WriteDecon.build_decon.decon_name
            Output prefix for 3dDeconvolve files

        Attributes
        ----------
        _xmat_path : path
            Location of X.<decon_name>.nocensor.xmat.1D

        Raises
        ------
        FileNotFoundError
            Missing expected x-file

        """
        # generate x-matrices
        self._xmat_path = os.path.join(
            self._subj_work, f"X.{decon_name}.nocensor.xmat.1D"
        )
        if os.path.exists(self._xmat_path):
            return

        # Execute decon_cmd, wait for singularity to close, and check
        print("\tRunning 3dDeconvolve for resting data")
        _, _ = submit.submit_sbatch(
            decon_cmd,
            f"dcn{self._subj[6:]}s{self._sess[-1]}",
            self._log_dir,
            mem_gig=6,
        )
        time.sleep(300)
        if not os.path.exists(self._xmat_path):
            raise FileNotFoundError(f"Expected to find {self._xmat_path}")

    def anaticor(
        self,
        decon_name,
        epi_masked,
        anat_dict,
        func_dict,
    ):
        """Project resting state correlation matrix.

        Parameters
        ----------
        decon_name : str, afni.WriteDecon.build_decon.decon_name
            Output prefix for 3dDeconvolve files
        epi_masked : path, afni.WriteDecon.build_decon.epi_masked
            Location of masked EPI data
        anat_dict : dict
            Contains reference names (key) and paths (value) to
            preprocessed anatomical files.
            Required keys:
            -   [mask-WMe] = path to eroded CSF mask
        func_dict : dict
            Contains reference names (key) and paths (value) to
            preprocessed functional files.
            Required keys:
            -   [func-scaled] = list of scaled EPI file paths
            -   [mot-cens] = path to censor vector

        Attributes
        ----------
        _reg_matrix : path
            Location of AFNI regression matrix

        Raises
        ------
        FileNotFoundError
            Missing regression matrix

        """
        # Check for x-matrix attribute
        if not hasattr(self, "_xmat_path"):
            raise AttributeError(
                "Attribute _xmat_path required, execute "
                + "ProjectRest.gen_xmatrix"
            )

        # Validate dict keys
        for _key in ["func-scaled", "mot-cens"]:
            if _key not in func_dict:
                raise KeyError(f"Expected {_key} key in func_dict")
        for _key in ["mask-WMe"]:
            if _key not in anat_dict:
                raise KeyError(f"Expected {_key} key in anat_dict")
        if len(func_dict["func-scaled"]) == 0:
            raise ValueError("Expected list of paths to scaled EPI files.")

        # Setup output path/name, avoid repeating work
        out_path = os.path.join(self._subj_work, f"{decon_name}_anaticor+tlrc")
        if os.path.exists(f"{out_path}.HEAD"):
            self._reg_matrix = out_path
            return out_path

        # Get WM signal
        print("\tProjecting correlation matrix")
        comb_path = os.path.join(self._subj_work, "tmp_epi-mask_WMe.nii.gz")
        if not os.path.exists(comb_path):
            bash_list = [
                "3dcalc",
                f"-a {epi_masked}",
                f"-b {anat_dict['mask-WMe']}",
                "-expr 'a*bool(b)'",
                "-datum float",
                f"-prefix {comb_path}",
            ]
            bash_cmd = " ".join(self._afni_prep + bash_list)
            _ = submit.submit_subprocess(bash_cmd, comb_path, "Comb mask")

        # Smooth WM signal
        blur_path = os.path.join(self._subj_work, "tmp_epi-blur.nii.gz")
        if not os.path.exists(blur_path):
            bash_list = [
                "3dmerge",
                "-1blur_fwhm 60",
                "-doall",
                f"-prefix {blur_path}",
                comb_path,
            ]
            bash_cmd = " ".join(self._afni_prep + bash_list)
            _ = submit.submit_subprocess(bash_cmd, blur_path, "Blur mask")

        # Project regression matrix, include WM as nuissance
        bash_list = [
            "3dTproject",
            "-polort 0",
            f"-input {func_dict['func-scaled'][0]}",
            f"-censor {func_dict['mot-cens']}",
            "-cenmode ZERO",
            f"-dsort {blur_path}",
            f"-ort {self._xmat_path}",
            f"-prefix {out_path}",
        ]
        bash_cmd = " ".join(self._afni_prep + bash_list)
        _, _ = submit.submit_sbatch(
            bash_cmd,
            f"pro{self._subj[6:]}s{self._sess[-1]}",
            self._log_dir,
            mem_gig=8,
        )
        time.sleep(300)

        # Check
        if not os.path.exists(f"{out_path}.HEAD"):
            raise FileNotFoundError(f"Expected to find {out_path}.HEAD")
        self._reg_matrix = out_path

    def seed_corr(self, anat_dict, coord_dict={"rPCC": "4 -54 24"}):
        """Project a seed-based correlation matrix.

        Construct a seed from coordinates, extract mean timeseries of seed,
        and use seed timeseries to project correlation matrix. Fisher
        Z-transfrom correlation matrix.

        Parameters
        ----------
        anat_dict : dict
            Contains reference names (key) and paths (value) to
            preprocessed anatomical files.
            Required keys:
            -   [mask-int] = path to intersection mask
        coord_dict : dict, optional
            keys = seed name, value = coordinate

        Returns
        -------
        dict
            Keys = seed names
            Values = path to Z-transformed correlation files

        Raises
        ------
        AttributeError
            Missing reg_matrix
        KeyError
            Missing mask-int key in anat_dict

        """
        # Check for attributes and keys
        if not hasattr(self, "_reg_matrix"):
            raise AttributeError(
                "Attribute reg_matrix required, execute ProjectRest.anaticor"
            )
        if "mask-int" not in anat_dict:
            raise KeyError("Expected key mask-int in anat_dict")

        # Generate seeds and get timeseries
        print("\tGenerating seed-based correlation matrices")
        seed_dict = self._coord_seed(coord_dict)

        # Find correlation with seeds, z-transform
        corr_dict = {}
        for seed, seed_ts in seed_dict.items():
            # Set output path/name, avoid repeating work
            print(f"\t\tWorking on seed {seed} ...")
            ztrans_file = self._reg_matrix.replace("+tlrc", f"_{seed}_ztrans")
            if os.path.exists(f"{ztrans_file}+tlrc.HEAD"):
                corr_dict[seed] = f"{ztrans_file}+tlrc.HEAD"
                continue

            # Correlation
            corr_file = self._reg_matrix.replace("+tlrc", f"_{seed}_corr")
            if not os.path.exists(f"{corr_file}+tlrc.HEAD"):
                bash_list = [
                    "3dTcorr1D",
                    f"-mask {anat_dict['mask-int']}",
                    f"-prefix {corr_file}",
                    self._reg_matrix,
                    seed_ts,
                ]
                bash_cmd = " ".join(self._afni_prep + bash_list)
                _ = submit.submit_subprocess(
                    bash_cmd, f"{corr_file}+tlrc.HEAD", "Corr mat"
                )

            # Transform
            bash_list = [
                "3dcalc",
                f"-a {corr_file}+tlrc",
                "-expr 'log((1+a)/(1-a))/2'",
                f"-prefix {ztrans_file}",
            ]
            bash_cmd = " ".join(self._afni_prep + bash_list)
            _ = submit.submit_subprocess(
                bash_cmd, f"{ztrans_file}+tlrc.HEAD", "Fisher Z"
            )
            corr_dict[seed] = f"{ztrans_file}+tlrc.HEAD"
        return corr_dict

    def _coord_seed(self, coord_dict, seed_radius=2):
        """Generate seed from coordinate and extract average timeseries.

        Parameters
        ----------
        coord_dict : dict
            keys = str, seed name
            value = str, coordinate
        seed_radius : int, optional
            Sphere radius (mm) from coordinate

        Returns
        -------
        dict
            Key = name of seed
            Value = path to averaged seed timeseries

        Raises
        ------
        AttributeError
            Missing attribute reg_matrix
        TypeError
            Improper setup of coord_dict

        """
        # Check coord_dict and attribute
        for name, coord in coord_dict.items():
            _coord_list = list(map(int, coord.split()))
            if not all(isinstance(x, int) for x in _coord_list):
                raise TypeError(
                    "Value of coord_dict must be space-separated integers"
                )
        if not hasattr(self, "_reg_matrix"):
            raise AttributeError(
                "Attribute reg_matrix required, execute ProjectRest.anaticor"
            )

        # Build seed and extract timeseries
        print("\t\tBuilding seeds")
        seed_dict = {}
        for seed, coord in coord_dict.items():
            # Avoid repeating work
            print(f"\t\t\tWorking on seed {seed}")
            seed_ts = os.path.join(
                self._subj_work, f"seed_{seed}_timeseries.1D"
            )
            if os.path.exists(seed_ts):
                seed_dict[seed] = seed_ts
                continue

            # Make seed
            seed_file = os.path.join(self._subj_work, f"seed_{seed}.nii.gz")
            tmp_coord = os.path.join(self._subj_work, f"tmp_seed_{seed}.txt")
            _ = submit.submit_subprocess(
                f"echo {coord} > {tmp_coord} ", tmp_coord, "Echo"
            )
            bash_list = [
                "3dUndump",
                f"-prefix {seed_file}",
                f"-master {self._reg_matrix}",
                f"-srad {seed_radius}",
                f"-xyz {tmp_coord}",
            ]
            bash_cmd = " ".join(self._afni_prep + bash_list)
            _ = submit.submit_subprocess(bash_cmd, seed_file, "Make seed")

            # Get average timeseries, deal with talkative singularity
            seed_long = os.path.join(
                self._subj_work, f"tmp_seed_{seed}_timeseries.1D"
            )
            bash_list = [
                "3dROIstats",
                "-quiet",
                f"-mask {seed_file}",
                f"{self._reg_matrix} > {seed_long}",
            ]
            bash_cmd = " ".join(self._afni_prep + bash_list)
            _ = submit.submit_subprocess(bash_cmd, seed_long, "Seed TS")
            _ = submit.submit_subprocess(
                f"tail -n +3 {seed_long} > {seed_ts}", seed_ts, "Tail"
            )
            seed_dict[seed] = seed_ts
        return seed_dict
