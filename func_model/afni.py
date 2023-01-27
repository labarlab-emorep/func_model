"""Methods for AFNI-based pipelines."""
# %%
import os
import json
import glob
import shutil
import time
import math
import statistics
import subprocess
import pandas as pd
import numpy as np
import nibabel as nib
from func_model import submit


# %%
def _prepend_afni_sing(proj_deriv, subj_work, sing_afni):
    """Supply singularity call for AFNI.

    Setup singularity call for AFNI in DCC EmoRep environment, used
    for prepending AFNI subprocess calls.

    Parameters
    ----------
    proj_deriv : path
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories
    subj_work : path
        Location of working directory for intermediates
    sing_afni : path
        Location of AFNI singularity file

    Returns
    -------
    list

    """
    return [
        "singularity run",
        "--cleanenv",
        f"--bind {proj_deriv}:{proj_deriv}",
        f"--bind {subj_work}:{subj_work}",
        f"--bind {subj_work}:/opt/home",
        sing_afni,
    ]


def valid_models(model_name):
    """Return bool of whether model_name is supported.

    Parameters
    ----------
    model_name : str
        [univ | rest | mixed]
        Desired AFNI model, for triggering different workflows

    Returns
    -------
    bool

    """
    valid_list = ["univ", "rest", "mixed"]
    return model_name in valid_list


class TimingFiles:
    """Create timing files for various types of EmoRep events.

    Aggregate all BIDS task events files for a participant's session,
    and then generate AFNI-style timing files (row number == run number)
    with onset or onset:duration information for each line.

    Timing files are written to:
        <subj_work>/timing_files

    Methods
    -------
    common_events(subj, sess, task)
        Generate and return timing files for common events
        (replay, judge, and wash)
    select_events(subj, sess, task)
        Generate and return timing files for selection trials
        (emotion, intensity)
    session_events(subj, sess, task)
        Generate and return timing files for movie or scenario emotions
    session_blocks(subj, sess, task)
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

    def __init__(self, subj_work, sess_events):
        """Initialize object.

        Setup attributes, make timing file directory, and combine all
        events files into a single dataframe.

        Parameters
        ----------
        subj_work : path
            Location of working directory for intermediates
        sess_events : list
            Paths to subject, session BIDS events files sorted
            by run number

        Attributes
        ----------
        _emo_switch : dict
            Switch for matching emotion names to AFNI length
        _sess_events : list
            Paths to subject, session BIDS events files sorted
            by run number
        _subj_tf_dir : path
            Output location for writing subject, session
            timing files

        Raises
        ------
        ValueError
            Unexpected task name
            Insufficient number of sess_events

        """
        print("\nInitializing TimingFiles")

        # Check arguments
        if len(sess_events) < 1:
            raise ValueError("Cannot make timing files from 0 events.tsv")

        # Set attributes, make output location
        self._sess_events = sess_events
        self._subj_tf_dir = os.path.join(subj_work, "timing_files")
        if not os.path.exists(self._subj_tf_dir):
            os.makedirs(self._subj_tf_dir)

        # Set switch for naming emotion timing files
        #   key = value in self._df_events["emotion"]
        #   value = AFNI-style description
        self._emo_switch = {
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

        # Generate dataframe from events files
        self._event_dataframe()

    def _event_dataframe(self):
        """Combine data from events files into dataframe.

        Attributes
        ----------
        _df_events : pd.DataFrame
            Column names == events files, run column added
        _events_run : list
            Run identifier extracted from event file name

        Raises
        ------
        ValueError
            The number of events files and number of runs are unequal

        """
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

    def common_events(self, subj, sess, task, marry=True, common_name=None):
        """Generate timing files for common events across both sessions.

        Make timing files for replay, judge, and wash events. Ouput timing
        file will use the 1D extension, use a BIDs-ish naming convention
        (sub-*_sess-*_task-*_desc-*_events.1D), and write the BIDS
        description field with "com[Rep|Jud|Was]" (e.g. desc-comRep).

        Fixations (fixS, fix) will be used as baseline in the deconvolution
        models, so timing files for these events are not generated.

        Parameters
        ----------
        subj : str
            BIDS subject identifier
        sess : str
            BIDS session identifier
        task : str
            [movies | scenarios]
            Name of task
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
            Incorrect task name

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

        # Validate task name
        valid_task = ["movies", "scenarios"]
        if task not in valid_task:
            raise ValueError(f"Inappropriate task name specified : {task}")

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
                self._subj_tf_dir,
                f"{subj}_{sess}_task-{task}_desc-{tf_name}_events.1D",
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

    def select_events(self, subj, sess, task, marry=True, select_name=None):
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
        subj : str
            BIDS subject identifier
        sess : str
            BIDS session identifier
        task : str
            [movies | scenarios]
            Name of task
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

        # Validate task name
        valid_task = ["movies", "scenarios"]
        if task not in valid_task:
            raise ValueError(f"Inappropriate task name specified : {task}")

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
                self._subj_tf_dir,
                f"{subj}_{sess}_task-{task}_desc-{tf_name}_events.1D",
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
        subj,
        sess,
        task,
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
        subj : str
            BIDS subject identifier
        sess : str
            BIDS session identifier
        task : str
            [movies | scenarios]
            Name of task
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

        # Validate task name
        valid_task = ["movies", "scenarios"]
        if task not in valid_task:
            raise ValueError(f"Inappropriate task name specified : {task}")

        # Provide emotions in sess_dict
        if emo_query:
            return [x for x in self._emo_switch.keys()]

        # Validate user input, generate emo_list
        if emotion_name:
            valid_list = [x for x in self._emo_switch.keys()]
            if emotion_name not in valid_list:
                raise ValueError(
                    f"Inappropriate emotion supplied : {emotion_name}"
                )
            emo_list = [emotion_name]
        else:

            # Identify unique emotions in dataframe
            trial_type_value = task[:-1]
            idx_sess = self._df_events.index[
                self._df_events["trial_type"] == trial_type_value
            ].tolist()
            emo_all = self._df_events.loc[idx_sess, "emotion"].tolist()
            emo_list = np.unique(np.array(emo_all)).tolist()

        # Generate timing files for all events in emo_list
        out_list = []
        for emo in emo_list:

            # Check that emo is found in planned dictionary
            if emo not in self.emo_switch.keys():
                raise ValueError(f"Unexpected emotion encountered : {emo}")

            # Determine timing file name, make an empty file
            # tf_name = (
            #     f"mov{self.emo_switch[emo]}"
            #     if task == "movies"
            #     else f"sce{self.emo_switch[emo]}"
            # )
            tf_name = task[:3] + self._emo_switch[emo]
            tf_path = os.path.join(
                self._subj_tf_dir,
                f"{subj}_{sess}_task-{task}_desc-{tf_name}_events.1D",
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

    def session_blocks(
        self,
        subj,
        sess,
        task,
    ):
        """Generate timing files for session-specific stimulus blocks.

        Ouput timing file will use the 1D extension, use a BIDs-ish naming
        convention (sub-*_sess-*_task-*_desc-*_events.1D), and write the BIDS
        description field with in a blk[Mov|Sce]Emotion format, using 3
        characters to identify the emotion. For example, blkMovFea = movie
        block with fear stimuli, blkSceCal = scenario with calmness stimuli.

        Parameters
        ----------
        subj : str
            BIDS subject identifier
        sess : str
            BIDS session identifier
        task : str
            [movies | scenarios]
            Name of task

        Returns
        -------
        list
            Paths to generated timing files

        Notes
        -----
        Timing files written to self._subj_tf_dir

        """
        # Validate task name
        valid_task = ["movies", "scenarios"]
        if task not in valid_task:
            raise ValueError(f"Inappropriate task name specified : {task}")

        # Identify unique emotions in dataframe
        trial_type_value = task[:-1]
        idx_sess = self._df_events.index[
            self._df_events["trial_type"] == trial_type_value
        ].tolist()
        emo_all = self._df_events.loc[idx_sess, "emotion"].tolist()
        emo_list = np.unique(np.array(emo_all)).tolist()

        # Generate timing files for all events in emo_list
        out_list = []
        block_dict = {}
        for emo in emo_list:

            # Check that emo is found in planned dictionary
            if emo not in self._emo_switch.keys():
                raise ValueError(f"Unexpected emotion encountered : {emo}")

            # Determine timing file name, make an empty file
            tf_name = "blk" + task.capitalize()[0] + self._emo_switch[emo]
            tf_path = os.path.join(
                self._subj_tf_dir,
                f"{subj}_{sess}_task-{task}_desc-{tf_name}_events.1D",
            )
            open(tf_path, "w").close()

            # Get emo info for each run
            dur_list = []
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
                    onset = self._df_events.loc[idx_emo, "onset"].tolist()
                    duration = self._df_events.loc[
                        idx_emo, "duration"
                    ].tolist()

                    block_onset = onset[0]
                    line_content = onset[0]
                    block_end = onset[-1] + duration[-1]
                    dur_list.append(block_end - block_onset)

                # Append line to timing file
                with open(tf_path, "a") as tf:
                    tf.writelines(f"{line_content}\n")

            # Add extra time for HRF duration
            block_dict[tf_name] = math.ceil(statistics.mean(dur_list)) + 14

            # Check for content in timing file
            if os.stat(tf_path).st_size == 0:
                raise RuntimeError(f"Empty file detected : {tf_path}")
            else:
                out_list.append(tf_path)

        # Write block_dict out to json
        out_json = os.path.join(self._subj_tf_dir, "block_durations.json")
        with open(out_json, "w") as jf:
            json.dump(block_dict, jf)
        return out_list


class MakeMasks:
    """Generate masks for AFNI-style analyses.

    Make masks required by, suggested for AFNI-style deconvolutions
    and group analyses.

    Methods
    -------
    intersect()
        Generate an anatomical-functional intersection mask
    tissue()
        Make eroded tissue masks
    minimum()
        Mask voxels with some meaningful signal across all
        volumes and runs.

    """

    def __init__(
        self,
        subj_work,
        proj_deriv,
        anat_dict,
        func_dict,
        sing_afni,
    ):
        """Initialize object.

        Parameters
        ----------
        subj_work : path
            Location of working directory for intermediates
        proj_deriv : path
            Location of project derivatives, containing fmriprep
            and fsl_denoise sub-directories
        anat_dict : dict
            Contains reference names (key) and paths (value) to
            preprocessed anatomical files.
            Required keys:
            -   [mask-brain] = path to fmriprep brain mask
            -   [mask-probCS] = path to fmriprep CSF label
            -   [mask-probWM] = path to fmriprep WM label
        func_dict : dict
            Contains reference names (key) and paths (value) to
            preprocessed functional files.
            Required keys:
            -   [func-preproc] = list of fmriprep preprocessed EPI paths
        sing_afni : path
            Location of AFNI singularity file

        Attributes
        ----------
        _sing_prep : list
            First part of subprocess call for AFNI singularity call
        _task : str
            BIDS task identifier

        Raises
        ------
        KeyError
            Missing expected key in anat_dict or func_dict

        """
        print("\nInitializing MakeMasks")

        # Validate dict keys
        for _key in ["mask-brain", "mask-probCS", "mask-probWM"]:
            if _key not in anat_dict:
                raise KeyError(f"Expected {_key} key in anat_dict")
        if "func-preproc" not in func_dict:
            raise KeyError("Expected func-preproc key in func_dict")

        # Set attributes
        self._subj_work = subj_work
        self._proj_deriv = proj_deriv
        self._anat_dict = anat_dict
        self._func_dict = func_dict
        self._sing_afni = sing_afni
        self._sing_prep = _prepend_afni_sing(
            self._proj_deriv, self._subj_work, self._sing_afni
        )

        try:
            _file_name = os.path.basename(func_dict["func-preproc"][0])
            subj, sess, task, _, _, _, _, _ = _file_name.split("_")
        except ValueError:
            raise ValueError(
                "BIDS file names required for items in func_dict: "
                + "subject, session, task, run, space, resolution, "
                + "description, and suffix.ext BIDS fields are "
                + "required by afni.MakeMasks. "
                + f"\n\tFound : {_file_name}"
            )
        self._subj = subj
        self._sess = sess
        self._task = task

    def intersect(self, c_frac=0.5, nbr_type="NN2", nbr_num=17):
        """Create an func-anat intersection mask.

        Generate a binary mask for voxels associated with both
        preprocessed anat and func data.

        Parameters
        ----------
        c_fract : float, optional
            Clip level fraction for AFNI's 3dAutomask
        nbr_type : str, optional
            [NN1 | NN2 | NN3]
            Nearest-neighbor type for AFNI's 3dAautomask
        nbr_num : int, optional
            Number of neibhors needed to avoid eroding in
            AFNI's 3dAutomask.

        Raises
        ------
        TypeError
            Invalid types for optional args
        ValueError
            Invalid parameters for optional args

        Returns
        -------
        path
            Location of anat-epi intersection mask

        """
        print("\tMaking intersection mask")

        # Validate arguments
        if not isinstance(c_frac, float):
            raise TypeError("c_frac must be float type")
        if not isinstance(nbr_type, str):
            raise TypeError("nbr_frac must be str type")
        if not isinstance(nbr_num, int):
            raise TypeError("nbr_numc must be int type")
        if c_frac < 0.1 or c_frac > 0.9:
            raise ValueError("c_fract must be between 0.1 and 0.9")
        if nbr_type not in ["NN1", "NN2", "NN3"]:
            raise ValueError("nbr_type must be NN1 | NN2 | NN3")
        if nbr_num < 6 or nbr_num > 26:
            raise ValueError("nbr_num must be between 6 and 26")

        # Setup output path, avoid repeating work
        out_path = (
            f"{self._subj_work}/{self._subj}_"
            + f"{self._sess}_{self._task}_desc-intersect_mask.nii.gz"
        )
        if os.path.exists(out_path):
            return out_path

        # Make binary masks for each preprocessed func file
        auto_list = []
        for run_file in self._func_dict["func-preproc"]:
            h_name = "tmp_autoMask_" + os.path.basename(run_file)
            h_out = os.path.join(self._subj_work, h_name)
            if not os.path.exists(h_out):
                bash_list = [
                    "3dAutomask",
                    f"-clfrac {c_frac}",
                    f"-{nbr_type}",
                    f"-nbhrs {nbr_num}",
                    f"-prefix {h_out}",
                    run_file,
                ]
                bash_cmd = " ".join(self._sing_prep + bash_list)
                _ = submit.submit_subprocess(bash_cmd, h_out, "Automask")
            auto_list.append(h_out)

        # Generate a union mask from the preprocessed masks
        union_out = os.path.join(
            self._subj_work, f"tmp_{self._task}_union.nii.gz"
        )
        if not os.path.exists(union_out):
            bash_list = [
                "3dmask_tool",
                f"-inputs {' '.join(auto_list)}",
                "-union",
                f"-prefix {union_out}",
            ]
            bash_cmd = " ".join(self._sing_prep + bash_list)
            _ = submit.submit_subprocess(bash_cmd, union_out, "Union")

        # Make anat-func intersection mask from the union and
        # fmriprep brain mask.
        bash_list = [
            "3dmask_tool",
            f"-input {union_out} {self._anat_dict['mask-brain']}",
            "-inter",
            f"-prefix {out_path}",
        ]
        bash_cmd = " ".join(self._sing_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_path, "Intersect")
        return out_path

    def tissue(self, thresh=0.5):
        """Make eroded tissue masks.

        Generate eroded white matter and CSF masks.

        Parameters
        ----------
        thresh : float, optional
            Threshold for binarizing probabilistic tissue
            mask output by fMRIPrep.

        Raises
        ------
        TypeError
            Inappropriate types for optional args
        ValueError
            Inappropriate value for optional args

        Returns
        -------
        dict
            ["CS"] = /path/to/eroded/CSF/mask
            ["WM"] = /path/to/eroded/WM/mask

        """
        # Validate args
        if not isinstance(thresh, float):
            raise TypeError("thresh must be float type")
        if thresh < 0.01 or thresh > 0.99:
            raise ValueError("thresh must be between 0.01 and 0.99")

        # Make CSF and WM masks
        out_dict = {"CS": "", "WM": ""}
        for tiss in out_dict.keys():
            print(f"\tMaking eroded tissue mask : {tiss}")

            # Setup final path, avoid repeating work
            out_tiss = os.path.join(
                self._subj_work,
                f"{self._subj}_{self._sess}_label-{tiss}e_mask.nii.gz",
            )
            if os.path.exists(out_tiss):
                out_dict[tiss] = out_tiss
                continue

            # Binarize probabilistic tissue mask
            in_path = self._anat_dict[f"mask-prob{tiss}"]
            bin_path = os.path.join(self._subj_work, f"tmp_{tiss}_bin.nii.gz")
            bash_list = [
                "c3d",
                in_path,
                f"-thresh {thresh} 1 1 0",
                f"-o {bin_path}",
            ]
            bash_cmd = " ".join(bash_list)
            _ = submit.submit_subprocess(
                bash_cmd, bin_path, f"Binarize {tiss}"
            )

            # Eroded tissue mask
            bash_list = [
                "3dmask_tool",
                f"-input {bin_path}",
                "-dilate_input -1",
                f"-prefix {out_tiss}",
            ]
            bash_cmd = " ".join(self._sing_prep + bash_list)
            _ = submit.submit_subprocess(bash_cmd, bin_path, f"Erode {tiss}")

            # Add path to eroded tissue mask
            out_dict[tiss] = out_tiss
        return out_dict

    def minimum(self):
        """Create a minimum-signal mask.

        Generate a mask for voxels in functional space that
        contain a value greater than some minimum threshold
        across all volumes and runs.

        Based around AFNI's 3dTstat -min.

        Returns
        -----
        path
            Location of minimum-value mask

        """
        print("\tMaking minimum value mask")

        # Setup file path, avoid repeating work
        out_path = (
            f"{self._subj_work}/{self._subj}_"
            + f"{self._sess}_{self._task}_desc-minval_mask.nii.gz"
        )
        if os.path.exists(out_path):
            return out_path

        # Make minimum value mask for each run
        min_list = []
        for run_file in self._func_dict["func-preproc"]:

            # Mask epi voxels that have some data
            h_name_bin = "tmp_bin_" + os.path.basename(run_file)
            h_out_bin = os.path.join(self._subj_work, h_name_bin)
            bash_list = [
                "3dcalc",
                "-overwrite",
                f"-a {run_file}",
                "-expr 1",
                f"-prefix {h_out_bin}",
            ]
            bash_cmd = " ".join(self._sing_prep + bash_list)
            _ = submit.submit_subprocess(bash_cmd, h_out_bin, "Binary EPI")

            # Make a mask for >min values
            h_name_min = "tmp_min_" + os.path.basename(run_file)
            h_out_min = os.path.join(self._subj_work, h_name_min)
            bash_list = [
                "3dTstat",
                "-min",
                f"-prefix {h_out_min}",
                h_out_bin,
            ]
            bash_cmd = " ".join(self._sing_prep + bash_list)
            min_list.append(
                submit.submit_subprocess(bash_cmd, h_out_min, "Minimum EPI")
            )

        # Average the minimum masks across runs
        h_name_mean = (
            f"tmp_{self._subj}_{self._sess}_{self._task}"
            + "_desc-mean_mask.nii.gz"
        )
        h_out_mean = os.path.join(self._subj_work, h_name_mean)
        bash_list = [
            "3dMean",
            "-datum short",
            f"-prefix {h_out_mean}",
            f"{' '.join(min_list)}",
        ]
        bash_cmd = " ".join(self._sing_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, h_out_mean, "Mean EPI")

        # Generate mask of non-zero voxels
        bash_list = [
            "3dcalc",
            f"-a {h_out_mean}",
            "-expr 'step(a-0.999)'",
            f"-prefix {out_path}",
        ]
        bash_cmd = " ".join(self._sing_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_path, "MinVal EPI")
        return out_path


def smooth_epi(
    subj_work,
    proj_deriv,
    func_preproc,
    sing_afni,
    blur_size=3,
):
    """Spatially smooth EPI files.

    Parameters
    ----------
    subj_work : path
        Location of working directory for intermediates
    proj_deriv : path
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories
    func_preproc : list
        Locations of preprocessed EPI files
    sing_afni : path
        Location of AFNI singularity file
    blur_size : int, optional
        Size (mm) of smoothing kernel

    Returns
    -------
    list
        Paths to smoothed EPI files

    Raises
    ------
    TypeError
        Improper parameter types

    """
    # Check arguments
    if not isinstance(blur_size, int):
        raise TypeError("Optional blur_size requires int")
    if not isinstance(func_preproc, list):
        raise TypeError("Argument func_preproc requires list")

    # Start return list, smooth each epi file
    print("\nSmoothing EPI files ...")
    func_smooth = []
    for epi_path in func_preproc:

        # Setup output names/paths, avoid repeating work
        epi_preproc = os.path.basename(epi_path)
        desc_preproc = epi_preproc.split("desc-")[1].split("_")[0]
        epi_smooth = epi_preproc.replace(desc_preproc, "smoothed")
        out_path = os.path.join(subj_work, epi_smooth)
        if os.path.exists(out_path):
            func_smooth.append(out_path)
            continue

        # Smooth data
        print(f"\tStarting smoothing of {epi_path}")
        bash_list = [
            "3dmerge",
            f"-1blur_fwhm {blur_size}",
            "-doall",
            f"-prefix {out_path}",
            epi_path,
        ]
        sing_prep = _prepend_afni_sing(proj_deriv, subj_work, sing_afni)
        bash_cmd = " ".join(sing_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_path, "Smooth run")

        # Update return list
        func_smooth.append(out_path)

    # Double-check correct order of files
    func_smooth.sort()
    return func_smooth


def scale_epi(subj_work, proj_deriv, mask_min, func_preproc, sing_afni):
    """Scale EPI timeseries.

    Parameters
    ----------
    subj_work : path
        Location of working directory for intermediates
    proj_deriv : path
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories
    mask_min : path
        Location of minimum-value mask, output of
        afni.MakeMasks.minimum
    func_preproc : list
        Locations of preprocessed EPI files
    sing_afni : path
        Location of AFNI singularity file

    Returns
    -------
    list
        Paths to scaled EPI files

    Raises
    ------
    TypeError
        Improper parameter types

    """
    # Check arguments
    if not isinstance(func_preproc, list):
        raise TypeError("Argument func_preproc requires list")

    # Start return list, scale each epi file supplied
    print("\nScaling EPI files ...")
    func_scaled = []
    for epi_path in func_preproc:

        # Setup output names, avoid repeating work
        epi_preproc = os.path.basename(epi_path)
        desc_preproc = epi_preproc.split("desc-")[1].split("_")[0]
        epi_tstat = "tmp_" + epi_preproc.replace(desc_preproc, "tstat")
        out_tstat = os.path.join(subj_work, epi_tstat)
        epi_scale = epi_preproc.replace(desc_preproc, "scaled")
        out_path = os.path.join(subj_work, epi_scale)
        if os.path.exists(out_path):
            func_scaled.append(out_path)
            continue

        # Determine mean values
        print(f"\tStarting scaling of {epi_path}")
        bash_list = [
            "3dTstat",
            f"-prefix {out_tstat}",
            epi_path,
        ]
        sing_prep = _prepend_afni_sing(proj_deriv, subj_work, sing_afni)
        bash_cmd = " ".join(sing_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_tstat, "Tstat run")

        # Scale values
        bash_list = [
            "3dcalc",
            f"-a {epi_path}",
            f"-b {out_tstat}",
            f"-c {mask_min}",
            "-expr 'c * min(200, a/b*100)*step(a)*step(b)'",
            f"-prefix {out_path}",
        ]
        bash_cmd = " ".join(sing_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_path, "Scale run")

        # Update return list
        func_scaled.append(out_path)

    # Double-check correct order of files
    func_scaled.sort()
    return func_scaled


class MotionCensor:
    """Make motion and censor files for AFNI deconvolution.

    Mine fMRIPrep timeseries.tsv files for required information,
    and generate files in a format AFNI can use in 3dDeconvolve.

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

    def __init__(self, subj_work, proj_deriv, func_motion, sing_afni):
        """Setup for making motion, censor files.

        Set attributes, make output directory, setup basic
        output file name.

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
        sing_afni : path
            Location of AFNI singularity file

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
        self._sing_afni = sing_afni
        self._sing_prep = _prepend_afni_sing(
            self._proj_deriv, self._subj_work, self._sing_afni
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


class WriteDecon:
    """Write an AFNI 3dDeconvolve command.

    Write 3dDeconvolve command supporting different basis functions
    and data types (task, resting-state).

    Attributes
    ----------
    decon_cmd : str
        Generated 3dDeconvolve command
    decon_name : str
        Prefix for output deconvolve files
    epi_masked : path
        Location of masked EPI data

    Methods
    -------
    build_decon(model_name)
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
        sing_afni,
    ):
        """Initialize object.

        Parameters
        ----------
        subj_work : path
            Location of working directory for intermediates
        proj_deriv : path
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
        sing_afni : path
            Location of AFNI singularity file

        Attributes
        ----------
        _afni_prep : list
            First part of subprocess call for AFNI singularity

        Raises
        ------
        KeyError
            Missing required keys in sess_func or sess_anat

        """
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
        self._func_dict = sess_func
        self._anat_dict = sess_anat
        self._sing_afni = sing_afni
        self._afni_prep = _prepend_afni_sing(
            self._proj_deriv, self._subj_work, self._sing_afni
        )

    def build_decon(self, model_name, sess_tfs=None):
        """Trigger deconvolution method.

        Use model_name to trigger the method the writes the
        relevant 3dDeconvolve command for the current pipeline.

        Parameters
        ----------
        model_name : str
            [univ | rest | mixed]
            Desired AFNI model, triggers right methods
        sess_tfs : None, dict, optional
            Required by model_name = univ|indiv.
            Contains reference names (key) and paths (value) to
            session AFNI-style timing files.

        Attributes
        ----------
        _tf_dict : dict, optional
            When model_name = univ | mixed
            Contains reference names (key) and paths (value) to
            session AFNI-style timing files.

        Raises
        ------
        ValueError
            Unsupported model name
            sess_tfs not supplied with univ, indiv model names

        """
        # Validate model name
        model_valid = valid_models(model_name)
        if not model_valid:
            raise ValueError(f"Unsupported model name : {model_name}")

        # Require timing files for task decons
        req_tfs = ["univ", "mixed"]
        if model_name in req_tfs:
            if not sess_tfs:
                raise ValueError(
                    f"Argument sess_tfs required with model_name={model_name}"
                )
            self._tf_dict = sess_tfs

        # Find, trigger appropriate method
        write_meth = getattr(self, f"write_{model_name}")
        write_meth()

    def _build_behavior(self, count_beh, basis_func):
        """Build a behavior regressor argument.

        Build a 3dDeconvolve behavior regressor accounting
        for desired basis function. Use with task deconvolutions
        (not resting-state pipelines).

        Parameters
        ----------
        count_beh : int
            On-going count to fill 3dDeconvolve -num_stimts
        basis_func : str
            [dur_mod | ind_mod]
            Desired basis function for behaviors in 3dDeconvolve

        Returns
        -------
        tuple
            [0] = behavior regressor for 3dDeconvolve
            [1] = count

        Raises
        ------
        ValueError
            Unsupported basis function

        """
        # Validate
        if basis_func == "two_gam":
            return ValueError("Basis function two_gam not currently supported")

        # Set inner functions for building different basis functions,
        # two_gam and tent are planned, ind_mod needs testing.
        def _beh_dur_mod(count_beh, tf_path):
            return f"-stim_times_AM1 {count_beh} {tf_path} 'dmBLOCK(1)'"

        def _beh_ind_mod(count_beh, tf_path):
            return f"-stim_times_IM {count_beh} {tf_path} 'dmBLOCK(1)'"

        # Map basis_func value to inner functions
        model_meth = {"dur_mod": _beh_dur_mod, "ind_mod": _beh_ind_mod}

        # Build regressor for each behavior
        print("\t\tBuilding behavior regressors ...")
        model_beh = []
        for tf_name, tf_path in self._tf_dict.items():
            count_beh += 1
            model_beh.append(model_meth[basis_func](count_beh, tf_path))
            model_beh.append(f"-stim_label {count_beh} {tf_name}")

        # Combine into string for 3dDeconvolve parameter
        reg_events = " ".join(model_beh)
        return (reg_events, count_beh)

    def write_univ(self, basis_func="dur_mod", decon_name="decon_univ"):
        """Write an AFNI 3dDeconvolve command for univariate checking.

        Build 3dDeconvolve command with minimal support for different
        basis functions.

        Parameters
        ----------
        basis_func : str, optional
            [dur_mod | ind_mod]
            Desired basis function for behaviors in 3dDeconvolve
        decon_name : str, optional
            Prefix for output deconvolve files

        Attributes
        ----------
        decon_cmd : str
            Generated 3dDeconvolve command
        decon_name : str
            Prefix for output deconvolve files

        Raises
        ------
        ValueError
            Unsupported basis_func value

        """
        # Validate
        if basis_func not in ["dur_mod", "ind_mod"]:
            raise ValueError("Invalid basis_func parameter")

        # Determine input variables for 3dDeconvolve
        print("\tBuilding 3dDeconvolve command ...")
        epi_preproc = " ".join(self._func_dict["func-scaled"])
        reg_motion_mean = self._func_dict["mot-mean"]
        reg_motion_deriv = self._func_dict["mot-deriv"]
        motion_censor = self._func_dict["mot-cens"]
        mask_int = self._anat_dict["mask-int"]

        # Build behavior regressors, get stimts count
        reg_events, count_beh = self._build_behavior(0, basis_func)

        # write decon command
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
            f"-num_stimts {count_beh}",
            reg_events,
            "-jobs 1",
            f"-x1D {self._subj_work}/X.{decon_name}.xmat.1D",
            f"-xjpeg {self._subj_work}/X.{decon_name}.jpg",
            f"-x1D_uncensored {self._subj_work}/X.{decon_name}.jpg",
            f"-bucket {self._subj_work}/{decon_name}_stats",
            f"-cbucket {self._subj_work}/{decon_name}_cbucket",
            f"-errts {self._subj_work}/{decon_name}_errts",
        ]
        decon_cmd = " ".join(self._afni_prep + decon_list)

        # Write script for review, records
        decon_script = os.path.join(self._subj_work, f"{decon_name}.sh")
        with open(decon_script, "w") as script:
            script.write(decon_cmd)
        self.decon_cmd = decon_cmd
        self.decon_name = decon_name

    def write_indiv(self):
        """Write an AFNI 3dDeconvolve command for individual mod checking.

        DEPRECATED.

        The "indiv" approach requires the same files and workflow as "univ",
        the only difference is in the basis function (and file name). So,
        use the class method write_univ with different parameters.

        """
        self.write_univ(basis_func="ind_mod", decon_name="decon_indiv")

    def write_mixed(self):
        """Write an AFNI 3dDeconvolve command for mixed modelling.

        Regressors include event and block designs.

        """
        pass

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
        epi_path = self._func_dict["func-scaled"][0]
        censor_path = self._func_dict["mot-cens"]
        reg_motion_mean = self._func_dict["mot-mean"]
        reg_motion_deriv = self._func_dict["mot-deriv"]

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
            self._func_dict["func-scaled"][0]
        )
        epi_masked = os.path.join(self._subj_work, mask_name)
        out_name = os.path.basename(self._func_dict["mot-cens"]).replace(
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
            f"-a {self._func_dict['func-scaled'][0]}",
            f"-b {self._anat_dict['mask-min']}",
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
            f"tmp_{os.path.basename(self._func_dict['mot-cens'])}",
        )
        bash_list = [
            "1d_tool.py",
            f"-set_run_lengths {epi_info['sum_vol']}",
            "-select_runs 1",
            f"-infile {self._func_dict['mot-cens']}",
            f"-write {out_cens}",
        ]
        bash_cmd = " ".join(self._afni_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_cens, "Cens rest")

        # Censor EPI data
        out_proj = os.path.join(
            self._subj_work,
            f"tmp_proj_{os.path.basename(self._func_dict['func-scaled'][0])}",
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
            f"-mask {self._anat_dict['mask-CSe']}",
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
                {self._func_dict["func-scaled"][0]} | \
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
        for epi_file in self._func_dict["func-scaled"]:

            # Extract number of volumes
            bash_cmd = f"""
                fslhd \
                    {self._func_dict["func-scaled"][0]} | \
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


class RunReml:
    """Run 3dREMLfit deconvolution.

    Setup for and execute 3dREMLfit command generated by
    3dDeconvolve (afni.WriteDecon.build_decon).

    Methods
    -------
    generate_reml(subj, sess, decon_cmd, decon_name)
        Execute 3dDeconvolve to generate 3dREMLfit command
    exec_reml(subj, sess, reml_path, decon_name)
        Setup for and run 3dREMLfit

    """

    def __init__(
        self,
        subj_work,
        proj_deriv,
        sess_anat,
        sess_func,
        sing_afni,
        log_dir,
    ):
        """Initialize object.

        Parameters
        ----------
        subj_work : path
            Location of working directory for intermediates
        proj_deriv : path
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
        sing_afni : path
            Location of AFNI singularity file
        log_dir : path
            Output location for log files and scripts

        Attributes
        ----------
        _afni_prep : list
            First part of subprocess call for AFNI singularity

        Raises
        ------
        KeyError
            sess_anat missing mask-WMe key
            sess_func missing func-scaled key

        """
        # Validate needed keys
        if "mask-WMe" not in sess_anat:
            raise KeyError("Expected mask-WMe key in sess_anat")
        if "func-scaled" not in sess_func:
            raise KeyError("Expected func-scaled key in sess_func")

        print("\nInitializing RunDecon")
        self._subj_work = subj_work
        self._sess_anat = sess_anat
        self._sess_func = sess_func
        self._sing_afni = sing_afni
        self._log_dir = log_dir
        self._afni_prep = _prepend_afni_sing(proj_deriv, subj_work, sing_afni)

    def generate_reml(self, subj, sess, decon_cmd, decon_name):
        """Generate matrices and 3dREMLfit command.

        Run the 3dDeconvolve command to generate the 3dREMLfit
        command and required input.

        Parameters
        ----------
        subj : str
            BIDS subject identifier
        sess : str
            BIDS session identifier
        decon_cmd : str, afni.WriteDecon.build_decon.decon_cmd
            Bash 3dDeconvolve command
        decon_name : str, afni.WriteDecon.build_decon.decon_name
            Output prefix for 3dDeconvolve files

        Returns
        -------
        path
            Location of 3dREMLfit script

        Raises
        ------
        ValueError
            Output 3dREMLfit script unexpected length

        """
        # Setup output file, avoid repeating work
        print("\tRunning 3dDeconvolve command ...")
        out_path = os.path.join(
            self._subj_work, f"{decon_name}_stats.REML_cmd"
        )
        if os.path.exists(out_path):
            return out_path

        # Execute decon_cmd, wait for singularity to close
        _, _ = submit.submit_sbatch(
            decon_cmd,
            f"dcn{subj[6:]}s{sess[-1]}",
            self._log_dir,
            mem_gig=10,
        )
        if not os.path.exists(out_path):
            time.sleep(300)

        # Check generated file length, account for 0 vs 1 indexing
        with open(out_path, "r") as rf:
            for line_count, _ in enumerate(rf):
                pass
        line_count += 1
        if line_count != 8:
            raise ValueError(
                f"Expected 8 lines in {out_path}, found {line_count}"
            )
        return out_path

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

    def exec_reml(self, subj, sess, reml_path, decon_name):
        """Exectue 3dREMLfit command.

        Setup for and exectue 3dREMLfit command generated
        by 3dDeconvolve. Writes reml command to:
            <subj_work>/decon_reml.sh

        Parameters
        ----------
        subj : str
            BIDS subject identfier
        sess : str
            BIDS session identifier
        reml_path : path, afni.WriteDecon.generate_reml
            Location of 3dREMLfit script
        decon_name : str
            Output prefix for 3dDeconvolve files

        Returns
        -------
        path
            Location of deconvolved HEAD file

        Raises
        ------
        ValueError
            Converting reml_path content to list failed

        """
        # Set final path name (anticipate AFNI output)
        out_path = reml_path.replace(".REML_cmd", "_REML+tlrc.HEAD")
        if os.path.exists(out_path):
            return out_path

        # Extract reml command from generated reml_path
        tail_path = os.path.join(self._subj_work, "decon_reml.txt")
        bash_cmd = f"tail -n 6 {reml_path} > {tail_path}"
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
                + f"contents of {reml_path} to a list"
            )

        # Make nuissance file, add to reml command
        nuiss_path = self._make_nuiss()
        reml_list.append(f"-dsort {nuiss_path}")
        reml_list.append("-GOFORIT")

        # Write script for review/records, then run
        print("\tRunning 3dREMLfit")
        bash_cmd = " ".join(self._afni_prep + reml_list)
        reml_script = os.path.join(self._subj_work, f"{decon_name}_reml.sh")
        with open(reml_script, "w") as script:
            script.write(bash_cmd)

        wall_time = 38 if decon_name == "decon_indiv" else 18
        _, _ = submit.submit_sbatch(
            bash_cmd,
            f"rml{subj[6:]}s{sess[-1]}",
            self._log_dir,
            num_hours=wall_time,
            num_cpus=6,
            mem_gig=8,
        )
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Expected to find {out_path}")
        return out_path


class ProjectRest:
    """Project a correlation matrix for resting state fMRI data.

    Execute generated 3ddeconvolve to produce a no-censor x-matrix,
    project correlation matrix accounting for WM and CSF nuissance,
    and conduct a seed-based correlation analysis.

    Methods
    -------
    gen_matrix(decon_cmd, decon_name)
        Execute 3ddeconvolve to make no-censor matrix
    anaticor(decon_name, epi_masked, anat_dict, func_dict)
        Generate regression matrix using anaticor method
    seed_corr(anat_dict)
        Generate seed-based correlation matrix

    """

    def __init__(self, subj, sess, subj_work, proj_deriv, sing_afni, log_dir):
        """Initialize object.

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

        Attributes
        ----------
        _afni_prep : list
            First part of subprocess call for AFNI singularity

        """
        print("\nInitializing ProjectRest")
        self._subj = subj
        self._sess = sess
        self._subj_work = subj_work
        self._log_dir = log_dir
        self._afni_prep = _prepend_afni_sing(
            proj_deriv, self._subj_work, sing_afni
        )

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
        if not hasattr(self, "xmat_path"):
            raise AttributeError(
                "Attribute xmat_path required, execute ProjectRest.gen_xmatrix"
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
        if not hasattr(self, "reg_matrix"):
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
        if not hasattr(self, "reg_matrix"):
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


class MoveFinal:
    """Copy final files from /work to /group.

    Identify desired files in /work and copy them
    to /group. Then purge /work.

    Methods
    -------
    copy_files(save_list)
        Copy list of files from /work to /group

    """

    def __init__(
        self, subj, sess, proj_deriv, subj_work, sess_anat, model_name
    ):
        """Copy files from work to group.

        Initiate object, construct list of desired files, then
        copy them to /group. Clean up /work.

        Parameters
        ----------
        subj : str
            BIDs subject identifier
        sess : str
            BIDs session identifier
        proj_deriv : path
            Location of project derivatives
        subj_work : path
            Location of working directory for intermediates
        sess_anat : dict
            Contains reference names (key) and paths (value) to
            preprocessed anatomical files.
            Required keys:
            -   [mask-WMe] = path to eroded CSF mask
            -   [mask-int] = path to intersection mask
        model_name : str
            Desired AFNI model, for triggering different workflows

        Raises
        ------
        KeyError
            Missing required key in sess_anat

        """
        # Validate dict keys
        for _key in ["mask-WMe", "mask-int"]:
            if _key not in sess_anat:
                raise KeyError(f"Expected {_key} key in sess_anat")

        # Set attributes
        self._subj = subj
        self._sess = sess
        self._proj_deriv = proj_deriv
        self._subj_work = subj_work
        self._sess_anat = sess_anat
        self._model_name = model_name

        # Trigger list construction, copy files
        save_list = (
            self._make_list_rest()
            if model_name == "rest"
            else self._make_list_task()
        )
        self.copy_files(save_list)

    def _make_list_task(self):
        """Find AFNI task files for saving.

        Files found:
        -   motion_files directory
        -   timing_files directory
        -   decon_<model_name>_stats_REML+tlrc.*
        -   decon_<model_name>.sh
        -   X.decon_<model_name>.*
        -   WM, intersection masks

        Returns
        -------
        list

        Raises
        ------
        FileNotFoundError
            decon_<model_name>_stats_REML+tlrc.* files were not found

        """
        subj_motion = os.path.join(self._subj_work, "motion_files")
        subj_timing = os.path.join(self._subj_work, "timing_files")
        stat_list = glob.glob(
            f"{self._subj_work}/decon_{self._model_name}_stats_REML+tlrc.*"
        )
        if stat_list:
            sh_list = glob.glob(
                f"{self._subj_work}/decon_{self._model_name}*.sh"
            )
            x_list = glob.glob(
                f"{self._subj_work}/X.decon_{self._model_name}.*"
            )
            save_list = stat_list + sh_list + x_list
            save_list.append(self._sess_anat["mask-WMe"])
            save_list.append(self._sess_anat["mask-int"])
            save_list.append(subj_motion)
            save_list.append(subj_timing)
        else:
            raise FileNotFoundError(
                f"Missing decon_{self._model_name} files in {self._subj_work}"
            )
        return save_list

    def _make_list_rest(self):
        """Find AFNI rest files for saving.

        Files found:
        -   decon_rest_anaticor+tlrc.*
        -   decon_rest.sh
        -   X.decon_rest.*
        -   Seed files
        -   Intersection mask

        Returns
        -------
        list

        Raises
        ------
        FileNotFoundError
            decon_rest_anaticor+tlrc.* files were not found

        """
        stat_list = glob.glob(f"{self._subj_work}/decon_rest_anaticor*+tlrc.*")
        if stat_list:
            seed_list = glob.glob(f"{self._subj_work}/seed_*")
            x_list = glob.glob(f"{self._subj_work}/X.decon_rest.*")
            save_list = stat_list + seed_list + x_list
            save_list.append(self._sess_anat["mask-int"])
            save_list.append(f"{self._subj_work}/decon_rest.sh")
        else:
            raise FileNotFoundError(
                f"Missing decon_rest files in {self._subj_work}"
            )
        return save_list

    def copy_files(self, save_list):
        """Copy desired files from /work to /group.

        Use bash subprocess of copy for speed, delete
        files in /work after copy has happened. Files
        are copied to:
            <proj_deriv>/model_afni/<subj>/<sess>/func

        Parameters
        ----------
        save_list : list
            Paths to files in /work that should be saved

        """
        # Setup save location in group directory
        subj_final = os.path.join(
            self._proj_deriv, "model_afni", self._subj, self._sess, "func"
        )
        if not os.path.exists(subj_final):
            os.makedirs(subj_final)

        # Copy desired files to group location
        for h_save in save_list:
            bash_cmd = f"cp -r {h_save} {subj_final}"
            h_sp = subprocess.Popen(
                bash_cmd, shell=True, stdout=subprocess.PIPE
            )
            _ = h_sp.communicate()
            h_sp.wait()
            chk_save = os.path.join(subj_final, h_save)
            if not os.path.exists(chk_save):
                raise FileNotFoundError(f"Expected to find {chk_save}")

        # Clean up - remove session directory in case
        # other session is still running.
        shutil.rmtree(os.path.dirname(self._subj_work))


# %%
def group_mask(proj_deriv, subj_list, out_dir):
    """Generate a group intersection mask.

    Make a union mask of all participant intersection masks. Output
    file is written to:
        <out_dir>/group_intersection_mask.nii.gz

    Parameters
    ----------
    proj_deriv : path
        Location of project derivatives directory
    subj_list : list
        Subjects to include in the mask
    out_dir : path
        Desired output location

    Returns
    -------
    path
        Location of group intersection mask

    Raises
    ------
    ValueError
        If not participant intersection masks are encountered

    """
    # Setup output path
    out_path = os.path.join(out_dir, "group_intersection_mask.nii.gz")
    if os.path.exists(out_path):
        return out_path

    # Identify intersection masks
    mask_list = []
    for subj in subj_list:
        for sess in ["ses-day2", "ses-day3"]:
            mask_path = os.path.join(
                proj_deriv,
                "model_afni",
                subj,
                sess,
                "func",
                f"{subj}_{sess}_desc-intersect_mask.nii.gz",
            )
            if os.path.exists(mask_path):
                mask_list.append(mask_path)
    if not mask_list:
        raise ValueError("Failed to find masks in model_afni")

    # Make group intersection mask
    bash_list = [
        "3dmask_tool",
        "-frac 1",
        f"-prefix {out_path}",
        f"-input {' '.join(mask_list)}",
    ]
    bash_cmd = " ".join(bash_list)
    _ = submit.submit_subprocess(bash_cmd, out_path, "Group Mask")
    return out_path


# %%
class ExtractTaskBetas:
    """Title.

    Desc.

    Attributes
    ----------

    Methods
    -------
    make_mask_matrix
    make_func_matrix
    comb_matrices

    """

    def __init__(self, proj_dir, out_dir, float_prec=4):
        """Initialize.

        Parameters
        ----------
        proj_dir : path
            Location of project directory
        out_dir : path
            Location of project output location
        float_prec : int, optional
            Desired float point precision of dataframes

        Raises
        ------
        TypeError
            Unexpected type for float_prec

        """
        if not isinstance(float_prec, int):
            raise TypeError("Expected float_prec type int")

        print("Initializing ExtractTaskBetas")
        self._proj_dir = proj_dir
        self.out_dir = out_dir
        self._float_prec = float_prec

    def _get_labels(self):
        """Get sub-brick levels from AFNI deconvolved file.

        Attributes
        ----------
        _stim_label : list
            Full label ID of sub-brick starting with "mov" or "sce",
            e.g. movFea#0_Coef.

        Raises
        ------
        ValueError
            Trouble parsing output of 3dinfo -label to list

        """
        print(f"\t\tGetting sub-brick labels for {self._subj}, {self._sess}")

        # Extract sub-brick label info
        out_label = os.path.join(self._subj_out_dir, "tmp_labels.txt")
        if not os.path.exists(out_label):
            bash_list = [
                "3dinfo",
                "-label",
                self._decon_path,
                f"> {out_label}",
            ]
            bash_cmd = " ".join(bash_list)
            _ = submit.submit_subprocess(bash_cmd, out_label, "Get labels")

        with open(out_label, "r") as lf:
            label_str = lf.read()
        label_list = [x for x in label_str.split("|")]
        if label_list[0] != "Full_Fstat":
            raise ValueError("Error in extracting decon labels.")

        # Identify labels relevant to task
        self._stim_label = [
            x
            for x in label_list
            if self._task.split("-")[1][:3] in x and "Fstat" not in x
        ]
        self._stim_label.sort()

    def _split_decon(self):
        """Split deconvolved files into files by sub-brick.

        Attributes
        ----------
        _beta_dict : dict
            key = emotion, value = path to sub-brick file

        Raises
        ------
        ValueError
            Sub-brick identifier int has length > 2

        """
        emo_switch = {
            "Amu": "amusement",
            "Ang": "anger",
            "Anx": "anxiety",
            "Awe": "awe",
            "Cal": "calmness",
            "Cra": "craving",
            "Dis": "disgust",
            "Exc": "excitement",
            "Fea": "fear",
            "Hor": "horror",
            "Joy": "joy",
            "Neu": "neutral",
            "Rom": "romance",
            "Sad": "sadness",
            "Sur": "surprise",
        }

        # Extract desired sub-bricks from deconvolve file by label name
        self._get_labels()
        beta_dict = {}
        for sub_label in self._stim_label:

            # Identify emo string, setup file name
            emo_long = emo_switch[sub_label[3:6]]
            out_file = (
                f"tmp_{self._subj}_{self._sess}_{self._task}_"
                + f"desc-{emo_long}_beta.nii.gz"
            )
            out_path = os.path.join(self._subj_out_dir, out_file)
            if os.path.exists(out_path):
                beta_dict[emo_long] = out_path
                continue

            # Determine sub-brick integer value
            print(f"\t\tExtracting sub-brick for {emo_long}")
            out_label = os.path.join(self._subj_out_dir, "tmp_label_int.txt")
            bash_list = [
                "3dinfo",
                "-label2index",
                sub_label,
                self._decon_path,
                f"> {out_label}",
            ]
            bash_cmd = " ".join(bash_list)
            _ = submit.submit_subprocess(bash_cmd, out_label, "Label int")

            with open(out_label, "r") as lf:
                label_int = lf.read().strip()
            if len(label_int) > 2:
                raise ValueError(f"Unexpected int length for {sub_label}")

            # Write sub-brick as new file
            bash_list = [
                "3dTcat",
                f"-prefix {out_path}",
                f"{self._decon_path}[{label_int}]",
                "> /dev/null 2>&1",
            ]
            bash_cmd = " ".join(bash_list)
            _ = submit.submit_subprocess(bash_cmd, out_label, "Split decon")
            beta_dict[emo_long] = out_path
        self._beta_dict = beta_dict

    def _flatten_array(self, arr: np.ndarray) -> np.ndarray:
        """Flatten 3D array and keep xyz index."""
        idx_val = []
        for x in np.arange(arr.shape[0]):
            for y in np.arange(arr.shape[1]):
                for z in np.arange(arr.shape[2]):
                    idx_val.append(
                        [
                            f"({x}, {y}, {z})",
                            round(arr[x][y][z], self._float_prec),
                        ]
                    )
        idx_val_arr = np.array(idx_val, dtype=object)
        return np.transpose(idx_val_arr)

    def _arr_to_df(self, arr: np.ndarray) -> pd.DataFrame:
        """Make dataframe from flat array."""
        df = pd.DataFrame(np.transpose(arr), columns=["idx", "val"])
        df = df.set_index("idx")
        df = df.transpose().reset_index(drop=True)
        return df

    def _nifti_to_arr(self, nifti_path: str) -> np.ndarray:
        """Generate flat array of NIfTI voxel values."""
        img = nib.load(nifti_path)
        img_data = img.get_fdata()
        img_flat = self._flatten_array(img_data)
        return img_flat

    def mask_coord(self, mask_path):
        """Find coordinates of voxels not in mask.

        Extract the voxel values from a binary mask

        Parameters
        ----------

        Attributes
        ----------
        _rm_cols

        """
        print("\tFinding coordinates to censor ...")
        img_flat = self._nifti_to_arr(mask_path)
        df_mask = self._arr_to_df(img_flat)
        self._rm_cols = df_mask.columns[df_mask.isin([0.0]).any()]

    def make_func_matrix(self, subj, sess, task, model_name, decon_path):
        """Title.

        Desc.

        Parameters
        ----------
        subj
        sess
        task
        model_name
        decon_path

        Attributes
        ----------
        _subj
        _sess
        _task
        _decon_path
        _subj_out_dir

        Returns
        -------
        pd.DataFrame

        """

        def _id_arr(emo: str) -> np.ndarray:
            """Return array of identifier information."""
            return np.array(
                [
                    ["subj_id", "task_id", "emo_id"],
                    [subj.split("-")[-1], task.split("-")[-1], emo],
                ]
            )

        # Set attributes
        if task not in ["task-scenarios", "task-movies"]:
            raise ValueError(f"Unexpected value for task : {task}")

        print(f"\tGetting betas from {subj}, {sess}")
        self._subj = subj
        self._sess = sess
        self._task = task
        self._decon_path = decon_path
        self._subj_out_dir = os.path.dirname(decon_path)

        #
        out_path = os.path.join(
            self._subj_out_dir,
            f"{subj}_{sess}_{task}_desc-{model_name}_betas.tsv",
        )
        if os.path.exists(out_path):
            return out_path

        #
        self._split_decon()

        #
        for emo, beta_path in self._beta_dict.items():
            print(f"\t\tExtracting betas for : {emo}")
            img_arr = self._nifti_to_arr(beta_path)
            info_arr = _id_arr(emo)
            img_arr = np.concatenate((info_arr, img_arr), axis=1)

            #
            if "df_betas" not in locals() and "df_betas" not in globals():
                df_betas = self._arr_to_df(img_arr)
            else:
                df_tmp = self._arr_to_df(img_arr)
                df_betas = pd.concat(
                    [df_betas, df_tmp], axis=0, ignore_index=True
                )
                del df_tmp
            del img_arr

        #
        print("\tCleaning dataframe ...")
        if hasattr(self, "_rm_cols"):
            df_betas = df_betas.drop(self._rm_cols, axis=1)

        #
        df_betas.to_csv(out_path, index=False, sep="\t")
        print(f"\t\tWrote : {out_path}")
        del df_betas
        tmp_list = glob.glob(f"{self._subj_out_dir}/tmp_*")
        for rm_file in tmp_list:
            os.remove(rm_file)
        return out_path

    def comb_matrices(self, subj_list, model_name, proj_deriv):
        """Title.

        Desc.

        Parameters
        ----------
        subj_list
        proj_deriv

        """
        print("\tCombining participant beta tsv files ...")
        df_list = sorted(
            glob.glob(
                f"{proj_deriv}/model_afni/**/*desc-{model_name}_betas.tsv",
                recursive=True,
            )
        )
        beta_list = [
            x
            for x in df_list
            if os.path.basename(x).split("_")[0] in subj_list
        ]

        #
        for beta_path in beta_list:
            print(f"\t\tAdding {beta_path} ...")
            if (
                "df_betas_all" not in locals()
                and "df_betas_all" not in globals()
            ):
                df_betas_all = pd.read_csv(beta_path, sep="\t")
            else:
                df_tmp = pd.read_csv(beta_path, sep="\t")
                df_betas_all = pd.concat(
                    [df_betas_all, df_tmp], axis=0, ignore_index=True
                )

        #
        out_path = os.path.join(
            self._proj_dir,
            "analyses/model_afni",
            f"afni_{model_name}_betas.tsv",
        )
        df_betas_all.to_csv(out_path, index=False, sep="\t")
        print(f"\tWrote : {out_path}")
        return out_path
