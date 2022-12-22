"""Methods for AFNI."""
import os
import json
import subprocess
import pandas as pd
import numpy as np
from func_model import submit


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


class TimingFiles:
    """Create timing files for various types of EmoRep events.

    Aggregate all BIDS task events files for a participant's session,
    and then generate AFNI-style timing files (row number == run number)
    with onset or onset:duration information for each line.

    Attributes
    ----------
    df_events : pd.DataFrame
        Combined events data into single dataframe
    events_run : list
        Run identifier extracted from event file name
    sess_events : list
        Paths to subject, session BIDS events files sorted
        by run number
    subj_tf_dir : path
        Output location for writing subject, session
        timing files

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
        sess_events : list
            Paths to subject, session BIDS events files sorted
            by run number
        subj_tf_dir : path
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
        Timing files written to self.subj_tf_dir

        """
        print("\tMaking timing files for common events")

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
                self.subj_tf_dir,
                f"{subj}_{sess}_task-{task}_desc-{tf_name}_events.1D",
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
        Timing files written to self.subj_tf_dir

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
                self.subj_tf_dir,
                f"{subj}_{sess}_task-{task}_desc-{tf_name}_events.1D",
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

    def session_events(
        self, subj, sess, task, marry=True, emotion_name=None, emo_query=False
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
        Timing files written to self.subj_tf_dir

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

        # Set emotion types
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
            trial_type_value = task[:-1]
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
                if task == "movies"
                else f"sce{sess_dict[emo]}"
            )
            tf_path = os.path.join(
                self.subj_tf_dir,
                f"{subj}_{sess}_task-{task}_desc-{tf_name}_events.1D",
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


class MakeMasks:
    """Generate masks for AFNI-style analyses.

    Make masks required by, suggested for AFNI-style deconvolutions
    and group analyses.

    Attributes
    ----------
    anat_dict : dict
        Contains reference names (key) and paths (value) to
        preprocessed anatomical files.
    func_dict : dict
        Contains reference names (key) and paths (value) to
        preprocessed functional files.
    proj_deriv : path
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories
    sess : str
        BIDS session identifier
    sing_afni : path
        Location of AFNI singularity file
    sing_prep : list
        First part of subprocess call for AFNI singularity
    subj : str
        BIDS subject identifier
    subj_work : path
        Location of working directory for intermediates

    Methods
    -------
    intersect(c_frac=0.5, nbr_type="NN2", n_nbr=17)
        Generate an anatomical-functional intersection mask
    tissue(thresh=0.5)
        Make eroded tissue masks
    minimum()
        Mask voxels with some meaningful signal across all
        volumes and runs.

    """

    def __init__(
        self,
        subj,
        sess,
        subj_work,
        proj_deriv,
        anat_dict,
        func_dict,
        sing_afni,
    ):
        """Initialize object.

        Parameters
        ----------
        subj : str
            BIDS subject identifier
        sess : str
            BIDS session identifier
        subj_work : path
            Location of working directory for intermediates
        proj_deriv : path
            Location of project derivatives, containing fmriprep
            and fsl_denoise sub-directories
        anat_dict : dict
            Contains reference names (key) and paths (value) to
            preprocessed anatomical files.
        func_dict : dict
            Contains reference names (key) and paths (value) to
            preprocessed functional files.
        sing_afni : path
            Location of AFNI singularity file

        Attributes
        ----------
        sing_prep : list
            First part of subprocess call for AFNI singularity

        """
        print("\nInitializing MakeMasks")
        self.subj = subj
        self.sess = sess
        self.subj_work = subj_work
        self.proj_deriv = proj_deriv
        self.anat_dict = anat_dict
        self.func_dict = func_dict
        self.sing_afni = sing_afni

        # Set singularity call
        self.sing_prep = _prepend_afni_sing(
            self.proj_deriv, self.subj_work, self.sing_afni
        )

    def intersect(self, c_frac=0.5, nbr_type="NN2", n_nbr=17):
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
        n_nbr : int, optional
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
        if not isinstance(n_nbr, int):
            raise TypeError("n_nbrc must be int type")
        if c_frac < 0.1 or c_frac > 0.9:
            raise ValueError("c_fract must be between 0.1 and 0.9")
        if nbr_type not in ["NN1", "NN2", "NN3"]:
            raise ValueError("nbr_type must be NN1 | NN2 | NN3")
        if n_nbr < 6 or n_nbr > 26:
            raise ValueError("n_nbr must be between 6 and 26")

        # Setup output path
        out_path = (
            f"{self.subj_work}/{self.subj}_"
            + f"{self.sess}_desc-intersect_mask.nii.gz"
        )
        if not os.path.exists(out_path):

            # Make binary masks for each preprocessed func file
            auto_list = []
            for run_file in self.func_dict["func-preproc"]:
                h_name = "tmp_autoMask_" + os.path.basename(run_file)
                h_out = os.path.join(self.subj_work, h_name)
                if not os.path.exists(h_out):
                    bash_list = [
                        "3dAutomask",
                        f"-clfrac {c_frac}",
                        f"-{nbr_type}",
                        f"-nbhrs {n_nbr}",
                        f"-prefix {h_out}",
                        run_file,
                    ]
                    bash_cmd = " ".join(self.sing_prep + bash_list)
                    _ = submit.submit_subprocess(bash_cmd, h_out, "Automask")
                auto_list.append(h_out)

            # Generate a union mask from the preprocessed masks
            union_out = os.path.join(self.subj_work, "tmp_union.nii.gz")
            if not os.path.exists(union_out):
                bash_list = [
                    "3dmask_tool",
                    f"-inputs {' '.join(auto_list)}",
                    "-union",
                    f"-prefix {union_out}",
                ]
                bash_cmd = " ".join(self.sing_prep + bash_list)
                _ = submit.submit_subprocess(bash_cmd, union_out, "Union")

            # Make anat-func intersection mask from the union and
            # fmriprep brain mask.
            bash_list = [
                "3dmask_tool",
                f"-input {union_out} {self.anat_dict['mask-brain']}",
                "-inter",
                f"-prefix {out_path}",
            ]
            bash_cmd = " ".join(self.sing_prep + bash_list)
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

            # Setup final path
            out_tiss = os.path.join(
                self.subj_work,
                f"{self.subj}_label-{tiss}e_mask.nii.gz",
            )
            if not os.path.exists(out_tiss):

                # Binarize probabilistic tissue mask
                in_path = self.anat_dict[f"mask-prob{tiss}"]
                bin_path = os.path.join(
                    self.subj_work, f"tmp_{tiss}_bin.nii.gz"
                )
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
                bash_cmd = " ".join(self.sing_prep + bash_list)
                _ = submit.submit_subprocess(
                    bash_cmd, bin_path, f"Erode {tiss}"
                )

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

        # Setup file path
        out_path = (
            f"{self.subj_work}/{self.subj}_"
            + f"{self.sess}_desc-minval_mask.nii.gz"
        )
        if not os.path.exists(out_path):

            # Make minimum value mask for each run
            min_list = []
            for run_file in self.func_dict["func-preproc"]:

                # Mask epi voxels that have some data
                h_name_bin = "tmp_bin_" + os.path.basename(run_file)
                h_out_bin = os.path.join(self.subj_work, h_name_bin)
                bash_list = [
                    "3dcalc",
                    "-overwrite",
                    f"-a {run_file}",
                    "-expr 1",
                    f"-prefix {h_out_bin}",
                ]
                bash_cmd = " ".join(self.sing_prep + bash_list)
                _ = submit.submit_subprocess(bash_cmd, h_out_bin, "Binary EPI")

                # Make a mask for >min values
                h_name_min = "tmp_min_" + os.path.basename(run_file)
                h_out_min = os.path.join(self.subj_work, h_name_min)
                bash_list = [
                    "3dTstat",
                    "-min",
                    f"-prefix {h_out_min}",
                    h_out_bin,
                ]
                bash_cmd = " ".join(self.sing_prep + bash_list)
                min_list.append(
                    submit.submit_subprocess(
                        bash_cmd, h_out_min, "Minimum EPI"
                    )
                )

            # Average the minimum masks across runs
            h_name_mean = f"tmp_{self.subj}_{self.sess}_desc-mean_mask.nii.gz"
            h_out_mean = os.path.join(self.subj_work, h_name_mean)
            bash_list = [
                "3dMean",
                "-datum short",
                f"-prefix {h_out_mean}",
                f"{' '.join(min_list)}",
            ]
            bash_cmd = " ".join(self.sing_prep + bash_list)
            _ = submit.submit_subprocess(bash_cmd, h_out_mean, "Mean EPI")

            # Generate mask of non-zero voxels
            bash_list = [
                "3dcalc",
                f"-a {h_out_mean}",
                "-expr 'step(a-0.999)'",
                f"-prefix {out_path}",
            ]
            bash_cmd = " ".join(self.sing_prep + bash_list)
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

        # Setup output names, paths
        epi_preproc = os.path.basename(epi_path)
        desc_preproc = epi_preproc.split("desc-")[1].split("_")[0]
        epi_smooth = epi_preproc.replace(desc_preproc, "smoothed")
        out_path = os.path.join(subj_work, epi_smooth)
        if not os.path.exists(out_path):

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

        # Setup output names
        epi_preproc = os.path.basename(epi_path)
        desc_preproc = epi_preproc.split("desc-")[1].split("_")[0]
        epi_tstat = "tmp_" + epi_preproc.replace(desc_preproc, "tstat")
        out_tstat = os.path.join(subj_work, epi_tstat)
        epi_scale = epi_preproc.replace(desc_preproc, "scaled")
        out_path = os.path.join(subj_work, epi_scale)
        if not os.path.exists(out_path):
            print(f"\tStarting scaling of {epi_path}")

            # Determine mean values
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

    Attributes
    ----------
    func_motion : list
        Locations of timeseries.tsv files produced by fMRIPrep
    proj_deriv : path
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories
    out_dir : path
        Output directory for motion, censor files
    out_str : str
        Basic output file name

    Methods
    -------
    mean_motion
        Make average motion file for 6 dof
    deriv_motion
        Make derivative motion file for 6 dof
    censor_volumes
        Make censor and inverted censor files
    count_motion
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

        Attributes
        ----------
        func_motion : list
            Locations of timeseries.tsv files produced by fMRIPrep
        proj_deriv : path
            Location of project derivatives, containing fmriprep
            and fsl_denoise sub-directories
        out_dir : path
            Output directory for motion, censor files
        out_str : str
            Basic output file name

        Raises
        ------
        TypeError
            Improper parameter types
        ValueError
            Improper naming convention of motion timeseries file

        """
        # Validate args
        if not isinstance(func_motion, list):
            raise TypeError("func_motion requires list type")

        print("\nInitializing MotionCensor")
        try:
            subj, sess, task, run, desc, suff = os.path.basename(
                func_motion[0]
            ).split("_")
        except ValueError:
            raise ValueError(
                "BIDS file names required for items in func_motion: "
                + "subject, session, task, run, description, and suffix.ext "
                + "BIDS fields are required by afni.MotionCensor."
            )
        self.out_str = f"{subj}_{sess}_{task}_{desc}_timeseries.1D"
        self.proj_deriv = proj_deriv
        self.func_motion = func_motion
        self.out_dir = os.path.join(subj_work, "motion_files")
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

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
            self.out_dir, self.out_str.replace("confounds", name)
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
        for mot_path in self.func_motion:
            df = pd.read_csv(mot_path, sep="\t")
            df_mean = df[labels_mean]
            df_mean = df_mean.round(6)
            mean_cat.append(df_mean)

        # Combine runs, write out
        df_mean_all = pd.concat(mean_cat, ignore_index=True)
        mean_out = self._write_df(df_mean_all, "mean")
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
        for mot_path in self.func_motion:
            df = pd.read_csv(mot_path, sep="\t")
            df_drv = df[labels_deriv]
            df_drv = df_drv.fillna(0)
            df_drv = df_drv.round(6)
            deriv_cat.append(df_drv)

        # Combine runs, write out
        df_deriv_all = pd.concat(deriv_cat, ignore_index=True)
        deriv_out = self._write_df(df_deriv_all, "deriv")
        return deriv_out

    def censor_volumes(self):
        """Make censor files.

        Generate AFNI-styled censor files, and also
        make inverted censor files to use as an inlusion
        mask. Volumes preceding motion events are
        censored.

        Attributes
        ----------
        df_censor : pd.DataFrame
            Row = volume
            Value = binary censor due to motion

        Returns
        -------
        tuple
            [0] = location of censor file
            [1] = location of inverted censor file

        """
        print("\tMaking censor and inverted censor files")

        # Setup and read-in data for runs
        censor_cat = []
        censor_inv = []
        for mot_path in self.func_motion:
            df = pd.read_csv(mot_path, sep="\t")

            # Combine motion events into new "sum" column
            df_cen = df.filter(regex="motion_outlier").copy()
            df_cen["sum"] = df_cen.sum(axis=1)
            df_cen = df_cen.astype(int)

            # Invert binary censor for regular AFNI censoring
            df_cen = df_cen.replace({0: 1, 1: 0})

            # Exclude volume preceding motion event
            zero_pos = df_cen.index[df_cen["sum"] == 0].tolist()
            zero_fill = [x - 1 for x in zero_pos]
            if -1 in zero_fill:
                zero_fill.remove(-1)
            df_cen.loc[zero_fill, "sum"] = 0
            censor_cat.append(df_cen)

            # Reinvert updated regression matrix for
            # inclusion mask.
            df_inv = df_cen.replace({0: 1, 1: 0})
            censor_inv.append(df_inv)

        # Combine run info, write out
        df_cen = pd.concat(censor_cat, ignore_index=True)
        censor_out = self._write_df(df_cen, "censor", ["sum"])
        df_cen_inv = pd.concat(censor_inv, ignore_index=True)
        censor_inv = self._write_df(df_cen_inv, "censorInv", ["sum"])
        self.df_censor = df_cen
        return (censor_out, censor_inv)

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
        -   Requires self.df_censor, will trigger method.
        -   Writes totals to self.out_dir/info_censored_volumes.json

        """
        print("\tCounting censored volumes")

        # Check for required attribute, trigger
        if not hasattr(self, "df_censor"):
            self.censor_volumes()

        # Quick calculations
        num_vol = self.df_censor["sum"].sum()
        num_tot = len(self.df_censor)
        cen_dict = {
            "total_volumes": int(num_tot),
            "included_volumes": int(num_vol),
            "proportion_excluded": round(1 - (num_vol / num_tot), 3),
        }

        # Write out and return
        out_path = os.path.join(self.out_dir, "info_censored_volumes.json")
        with open(out_path, "w") as jfile:
            json.dump(cen_dict, jfile)
        return out_path


class WriteDecon:
    """Write an AFNI 3dDeconvolve command.

    Write 3dDeconvolve and then execute it to generate
    matrices and 3dREMLfit script.

    Attributes
    ----------
    afni_prep : list
        First part of subprocess call for AFNI singularity
    anat_dict : dict
        Contains reference names (key) and paths (value) to
        preprocessed anatomical files
    decon_cmd : str
        Generated 3dDeconvolve command
    decon_name : str
        Prefix for output deconvolve files
    func_dict : dict
        Contains reference names (key) and paths (value) to
        preprocessed functional files
    proj_deriv : path
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories
    sing_afni : path
        Location of AFNI singularity file
    subj_work : path
        Location of working directory for intermediates
    tf_dict : dict
        Contains reference names (key) and paths (value) to
        session AFNI-style timing files

    Methods
    -------
    build_decon(model_name: str)
        Trigger the appropriate method for the current pipeline, e.g.
        build_decon(model_name="univ") causes the method "write_univ"
        to be executed.
    write_univ(basis_func: str = "dur_mod, decon_name: str = "decon_univ")
        Write a univariate 3dDeconvolve command for sanity checks
    generate_reml(log_dir: path)
        Execute 3dDeconvolve to generate 3dREMLfit command

    """

    def __init__(
        self,
        subj_work,
        proj_deriv,
        sess_func,
        sess_anat,
        sess_tfs,
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
        sess_func : dict
            Contains reference names (key) and paths (value) to
            preprocessed functional files
        sess_anat : dict
            Contains reference names (key) and paths (value) to
            preprocessed anatomical files
        sess_tfs : dict
            Contains reference names (key) and paths (value) to
            session AFNI-style timing files
        sing_afni : path
            Location of AFNI singularity file

        Attributes
        ----------
        afni_prep : list
            First part of subprocess call for AFNI singularity

        """
        print("\nInitializing WriteDecon")
        self.proj_deriv = proj_deriv
        self.subj_work = subj_work
        self.func_dict = sess_func
        self.anat_dict = sess_anat
        self.tf_dict = sess_tfs
        self.sing_afni = sing_afni

        # Start singulartiy call
        self.afni_prep = _prepend_afni_sing(
            self.proj_deriv, self.subj_work, self.sing_afni
        )

    def build_decon(self, model_name):
        """Trigger deconvolution method.

        Use model_name to trigger the method the writes the
        relevant 3dDeconvolve command for the current pipeline.

        Parameters
        ----------
        model_name : str
            [univ]
            Desired AFNI model, triggers write methods

        Raises
        ------
        ValueError
            Unsupported model name

        """
        valid_names = ["univ"]
        if model_name not in valid_names:
            raise ValueError(f"Unsupported model name : {model_name}")

        write_meth = getattr(self, f"write_{model_name}")
        write_meth()

    def _build_censor_ts(self, count_beh):
        """Make a censor regressor argument.

        Build a classic HRF and convolve it with the censor vector
        to simulate change in BOLD from motion. This is done instead
        of removing the motion volumes in order to keep a
        continuous timeseries.

        Parameters
        ----------
        count_beh : int
            On-going count to fill 3dDeconvolve -num_stimts

        Returns
        -------
        tuple
            [0] = censor regressor for 3dDeconvolve
            [1] = count

        Raises
        ------
        KeyError
            func_dict missing func-inv key

        """
        # Validate reqs
        if "func-inv" not in self.func_dict:
            raise KeyError("Expected func-inv key in func_dict")

        # Setup output path and name
        print("\t\tBuilding censor waveform ...")
        out_name = os.path.basename(self.func_dict["func-inv"]).replace(
            "Inv_timeseries", "_HRF"
        )
        out_path = os.path.join(self.subj_work, out_name)

        # Get session info, for TR length, number of volumes
        sess_info = self._get_sess_info()

        # Generate classic HRF
        len_tr = sess_info["len_tr"]
        bash_list = [
            "3dDeconvolve",
            "-polort -1",
            f"-nodata 10 {len_tr}",
            "-num_stimts 1",
            f"-stim_times 1 1D:0 'BLOCK({len_tr}, 1)'",
            f"-x1D {self.subj_work}/Classic_HRF.1D",
            "-x1D_stop",
        ]
        bash_cmd = " ".join(self.afni_prep + bash_list)
        _ = submit.submit_subprocess(
            bash_cmd, f"{self.subj_work}/Classic_HRF.1D", "Classic HRF"
        )

        # Convolve HRF with censor file
        waver_out = os.path.join(self.subj_work, "tmp_waver.1D")
        bash_list = [
            "waver",
            f"-FILE {len_tr} {self.subj_work}/Classic_HRF.1D",
            "-peak 1",
            f"-input {self.func_dict['func-inv']}",
            f"-numout {sess_info['sum_vol']}",
            f"> {self.subj_work}/tmp_waver.1D",
        ]
        bash_cmd = " ".join(self.afni_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, waver_out, "Censor TS temp")

        # Manage talkative singularity
        bash_cmd = f"tail -n {sess_info['sum_vol']} {waver_out} > {out_path}"
        _ = submit.submit_subprocess(bash_cmd, out_path, "Censor TS")

        # Make baseline regressor argument
        count_beh += 1
        reg_censor_list = [
            f"-stim_file {count_beh} {out_path}",
            f"-stim_base {count_beh}",
            f"-stim_label {count_beh} mot_cens",
        ]
        reg_censor = " ".join(reg_censor_list)

        return (reg_censor, count_beh)

    def _get_sess_info(self):
        """Return dict of TR, volume info.

        Returns
        -------
        dict

        Raises
        ------
        KeyError
            Missing func-scaled key in func_dict
        ValueError
            No value in func_dict[func-scaled]

        """
        # Check for EPI data
        if "func-scaled" not in self.func_dict:
            raise KeyError("Expected func-scaled key in func_dict")
        if len(self.func_dict["func-scaled"]) < 1:
            raise ValueError(
                "Expected at least one value in func_dict[func-scaled]"
            )

        # Find TR length
        bash_cmd = f"""
            fslhd \
                {self.func_dict["func-scaled"][0]} | \
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
        for epi_file in self.func_dict["func-scaled"]:

            # Extract number of volumes
            bash_cmd = f"""
                fslhd \
                    {self.func_dict["func-scaled"][0]} | \
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

    def _build_behavior(self, count_beh, basis_func):
        """Build a behavior regressor argument

        Build a 3dDeconvolve behavior regressor accounting
        for desired basis function.

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
        for tf_name, tf_path in self.tf_dict.items():
            count_beh += 1
            model_beh.append(model_meth[basis_func](count_beh, tf_path))
            model_beh.append(f"-stim_label {count_beh} {tf_name}")

        # Combine into string for 3dDeconvolve parameter
        reg_events = " ".join(model_beh)
        return (reg_events, count_beh)

    def write_univ(self, basis_func="dur_mod", decon_name="decon_univ"):
        """Write an AFNI 3dDeconvolve command for sanity checking.

        Build 3dDeconvolve command with minimal support for different
        basis functions.

        Parameters
        ----------
        basis_func : str, optional
            [dur_mod | ind_mod | two_gam]
            Desired basis function for behaviors in 3dDeconvolve
        decon_name : str, optional
            Prefix for output deconvolve files

        Attributes
        ----------
        decon_name : str
            Prefix for output deconvolve files
        decon_cmd : str
            Generated 3dDeconvolve command

        Raises
        ------
        ValueError
            Unsupported basis_func value

        """
        # Validate
        valid_list = ["dur_mod", "ind_mod", "two_gam"]
        if basis_func not in valid_list:
            raise ValueError("Invalid basis_func parameter")

        for key in ["func-scaled", "func-mean", "func-deriv"]:
            if key not in self.func_dict:
                raise KeyError(f"Expected {key} key in func_dict")

        # Set variables and paths
        print("\tBuilding 3dDeconvolve command ...")
        self.decon_name = decon_name
        epi_preproc = " ".join(self.func_dict["func-scaled"])
        reg_motion_mean = self.func_dict["func-mean"]
        reg_motion_deriv = self.func_dict["func-deriv"]

        # Start counter for num_stimts, build regressors
        count_beh = 0
        reg_censor, count_beh = self._build_censor_ts(count_beh)
        reg_events, count_beh = self._build_behavior(count_beh, basis_func)

        # write decon command
        decon_list = [
            "3dDeconvolve",
            "-x1D_stop",
            "-GOFORIT",
            f"-input {epi_preproc}",
            f"-ortvec {reg_motion_mean} mot_mean",
            f"-ortvec {reg_motion_deriv} mot_deriv",
            "-polort A",
            "-float",
            "-local_times",
            f"-num_stimts {count_beh}",
            reg_censor,
            reg_events,
            "-jobs 1",
            f"-x1D {self.subj_work}/X.{decon_name}.xmat.1D",
            f"-xjpeg {self.subj_work}/X.{decon_name}.jpg",
            f"-x1D_uncensored {self.subj_work}/X.{decon_name}.jpg",
            f"-bucket {self.subj_work}/{decon_name}_stats",
            f"-cbucket {self.subj_work}/{decon_name}_cbucket",
            f"-errts {self.subj_work}/{decon_name}_errts",
        ]
        self.decon_cmd = " ".join(self.afni_prep + decon_list)

        # Write script for review, records
        decon_script = os.path.join(self.subj_work, f"{decon_name}.sh")
        with open(decon_script, "w") as script:
            script.write(self.decon_cmd)

    def generate_reml(self, subj, sess, log_dir):
        """Generate matrices and 3dREMLfit command.

        Run the 3dDeconvolve command to generate the 3dREMLfit
        command and required input.

        Parameters
        ----------
        subj : str
            BIDS subject identifier
        sess : str
            BIDS session identifier
        log_dir : path
            Output location for logs

        Returns
        -------
        path
            Location of 3dREMLfit script

        Raises
        ------
        AttributeError
            Missing decon_cmd
            Missing decon_name
        ValueError
            Output 3dREMLfit script unexpected length

        """
        # Check for required attributes
        if not hasattr(self, "decon_cmd"):
            raise AttributeError(
                "No decon_cmd detected, try WriteDecon.write_univ."
            )
        if not hasattr(self, "decon_name"):
            raise AttributeError(
                "No decon_name detected, try WriteDecon.write_univ."
            )

        # Execute decon_cmd
        print("\tRunning 3dDeconvolve command ...")
        out_path = os.path.join(
            self.subj_work, f"{self.decon_name}_stats.REML_cmd"
        )
        if not os.path.exists(out_path):
            _, _ = submit.submit_sbatch(
                self.decon_cmd,
                f"dcn{subj[6:]}s{sess[-1]}",
                log_dir,
                mem_gig=10,
            )

        # Check generated file length
        with open(out_path, "r") as rf:
            for line_count, _ in enumerate(rf):
                pass
        if line_count + 1 != 7:
            raise ValueError(f"Expected 7 lines in {out_path}")
        return out_path


class RunDecon:
    """Run 3dREMLfit deconvolution.

    Setup for and execute 3dREMLfit command generated by
    3dDeconvolve (afni.WriteDecon.generate_decon_<foo>).

    Attributes
    ----------
    afni_prep : list
        First part of subprocess call for AFNI singularity
    log_dir : path
        Output location for log files and scripts
    sess_anat : dict
        Contains reference names (key) and paths (value) to
        preprocessed anatomical files
    sess_func : dict
        Contains reference names (key) and paths (value) to
        preprocessed functional files
    sing_afni : path
        Location of AFNI singularity file
    subj : str
        BIDS subject identifier
    subj_work : path
        Location of working directory for intermediates
    reml_path : path
        Output of afni.WriteDecon.generate_decon_<foo>,
        location of foo.REML_cmd file written by 3ddeconvolve

    Methods
    -------
    exec_reml
        Setup for and run 3dREMLfit

    """

    def __init__(
        self,
        subj_work,
        proj_deriv,
        reml_path,
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
            and fsl_denoise sub-directories
        reml_path : path
            Output of afni.WriteDecon.generate_decon_<foo>,
            location of foo.REML_cmd file written by 3ddeconvolve
        sess_anat : dict
            Contains reference names (key) and paths (value) to
            preprocessed anatomical files
        sess_func : dict
            Contains reference names (key) and paths (value) to
            preprocessed functional files
        sing_afni : path
            Location of AFNI singularity file
        log_dir : path
            Output location for log files and scripts

        Attributes
        ----------
        afni_prep : list
            First part of subprocess call for AFNI singularity

        """
        print("\nInitializing RunDecon")

        # Capture parameters
        self.subj_work = subj_work
        self.reml_path = reml_path
        self.sess_anat = sess_anat
        self.sess_func = sess_func
        self.sing_afni = sing_afni
        self.log_dir = log_dir

        # Set attributes, trigger reml execution
        self.afni_prep = _prepend_afni_sing(proj_deriv, subj_work, sing_afni)

    def _make_nuiss(self):
        """Make noise estimation file.

        Generate file containing white matter EPI signal, used with
        -dsort option in 3dREMLfit.

        Returns
        -------
        path
            Location of WM signal file

        Raises
        ------
        KeyError
            sess_anat missing mask-WMe key
            sess_func missing func-scaled key

        """
        print("\tMaking nuissance mask")

        # Validate needed keys
        if "mask-WMe" not in self.sess_anat:
            raise KeyError("Expected mask-WMe key in sess_anat")
        if "func-scaled" not in self.sess_func:
            raise KeyError("Expected func-scaled key in sess_func")

        # Setup output location and name
        h_name = os.path.basename(self.sess_anat["mask-WMe"])
        out_path = os.path.join(
            self.subj_work, h_name.replace("label-WMe", "desc-nuiss")
        )
        if not os.path.exists(out_path):

            # Concatenate EPI runs
            out_tcat = os.path.join(self.subj_work, "tmp_tcat_all-runs.nii.gz")
            bash_list = [
                "3dTcat",
                f"-prefix {out_tcat}",
                " ".join(self.sess_func["func-scaled"]),
            ]
            bash_cmd = " ".join(self.afni_prep + bash_list)
            _ = submit.submit_subprocess(bash_cmd, out_tcat, "Tcat runs")

            # Make eroded mask in EPI time
            out_erode = os.path.join(
                self.subj_work, "tmp_eroded_all-runs.nii.gz"
            )
            bash_list = [
                "3dcalc",
                f"-a {out_tcat}",
                f"-b {self.sess_anat['mask-WMe']}",
                "-expr 'a*bool(b)'",
                "-datum float",
                f"-prefix {out_erode}",
            ]
            bash_cmd = " ".join(self.afni_prep + bash_list)
            _ = submit.submit_subprocess(bash_cmd, out_erode, "Erode")

            # Generate blurred WM file
            bash_list = [
                "3dmerge",
                "-1blur_fwhm 20",
                "-doall",
                f"-prefix {out_path}",
                out_erode,
            ]
            bash_cmd = " ".join(self.afni_prep + bash_list)
            _ = submit.submit_subprocess(bash_cmd, out_erode, "Erode")

        return out_path

    def exec_reml(self, subj, sess):
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
        out_path = self.reml_path.replace(".REML_cmd", "_REML+tlrc.HEAD")
        if os.path.exists(out_path):
            return out_path

        # Extract reml command from generated reml_path
        tail_path = os.path.join(self.subj_work, "decon_reml.txt")
        bash_cmd = f"tail -n 5 {self.reml_path} > {tail_path}"
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
                + f"contents of {self.reml_path} to a list"
            )

        # Make nuissance file, add to reml command
        nuiss_path = self._make_nuiss()
        reml_list.append(f"-dsort {nuiss_path}")
        reml_list.append("-GOFORIT")

        # Write script for review/records, then run
        print("\tRunning 3dREMLfit")

        bash_cmd = " ".join(self.afni_prep + reml_list)
        reml_script = os.path.join(self.subj_work, "decon_reml.sh")
        with open(reml_script, "w") as script:
            script.write(bash_cmd)

        _, _ = submit.submit_sbatch(
            bash_cmd,
            f"rml{subj[6:]}s{sess[-1]}",
            self.log_dir,
            num_hours=20,
            num_cpus=6,
            mem_gig=8,
        )
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Expected to find {out_path}")
        return out_path
