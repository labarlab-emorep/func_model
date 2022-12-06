"""Methods for AFNI."""
import os
import pandas as pd
import numpy as np
from func_model import submit


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
        Paths to subject, session BIDS events files sorted
        by run number

    Attributes
    ----------
    df_events : pd.DataFrame
        Combined events data into single dataframe
    events_run : list
        Run identifier extracted from event file name
    sess : str
        BIDS session identifier
    sess_events : list
        Paths to subject, session BIDS events files sorted
        by run number
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
            Paths to subject, session BIDS events files sorted
            by run number

        Attributes
        ----------
        sess : str
            BIDS session identifier
        sess_events : list
            Paths to subject, session BIDS events files sorted
            by run number
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
        print("Making timing files for common events")

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
        print("Making timing files for selection trials")

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
        print("Making timing files for session-specific trials")

        # Validate bool args
        if not isinstance(emo_query, bool):
            raise TypeError("Argument 'emo_query' is bool")
        if not isinstance(marry, bool):
            raise TypeError("Argument 'marry' is bool")

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

        """
        self.subj = subj
        self.sess = sess
        self.subj_work = subj_work
        self.proj_deriv = proj_deriv
        self.anat_dict = anat_dict
        self.func_dict = func_dict
        self.sing_afni = sing_afni

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

        Notes
        -----
        Adds new key:value to self.anat_dict
            ["mask-intersect"] = /path/to/intersection/mask

        """
        print("Making intersection mask")

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
                        "singularity run",
                        "--cleanenv",
                        f"--bind {self.proj_deriv}:{self.proj_deriv}",
                        f"--bind {self.subj_work}:{self.subj_work}",
                        f"--bind {self.subj_work}:/opt/home",
                        self.sing_afni,
                        "3dAutomask",
                        f"-clfrac {c_frac}",
                        f"-{nbr_type}",
                        f"-nbhrs {n_nbr}",
                        f"-prefix {h_out}",
                        run_file,
                    ]
                    bash_cmd = " ".join(bash_list)
                    _ = submit.submit_subprocess(bash_cmd, h_out, "Automask")
                auto_list.append(h_out)

            # Generate a union mask from the preprocessed masks
            union_out = os.path.join(self.subj_work, "tmp_union.nii.gz")
            if not os.path.exists(union_out):
                bash_list = [
                    "singularity run",
                    "--cleanenv",
                    f"--bind {self.proj_deriv}:{self.proj_deriv}",
                    f"--bind {self.subj_work}:{self.subj_work}",
                    f"--bind {self.subj_work}:/opt/home",
                    self.sing_afni,
                    "3dmask_tool",
                    f"-inputs {' '.join(auto_list)}",
                    "-union",
                    f"-prefix {union_out}",
                ]
                bash_cmd = " ".join(bash_list)
                _ = submit.submit_subprocess(bash_cmd, union_out, "Union")

            # Make anat-func intersection mask from the union and
            # fmriprep brain mask.
            bash_list = [
                "singularity run",
                "--cleanenv",
                f"--bind {self.proj_deriv}:{self.proj_deriv}",
                f"--bind {self.subj_work}:{self.subj_work}",
                f"--bind {self.subj_work}:/opt/home",
                self.sing_afni,
                "3dmask_tool",
                f"-input {union_out} {self.anat_dict['mask-brain']}",
                "-inter",
                f"-prefix {out_path}",
            ]
            bash_cmd = " ".join(bash_list)
            _ = submit.submit_subprocess(bash_cmd, out_path, "Intersect")

        # Add path to intersection mask to attribute
        self.anat_dict["mask-intersect"] = out_path

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

        Notes
        -----
        Adds new key:value to self.anat_dict
            ["mask-CSe"] = /path/to/eroded/CSF/mask
            ["mask-WMe"] = /path/to/eroded/WM/mask

        """
        # Validate args
        if not isinstance(thresh, float):
            raise TypeError("thresh must be float type")
        if float < 0.01 or float > 0.99:
            raise ValueError("thresh must be between 0.01 and 0.99")

        # Make CSF and WM masks
        for tiss in ["CS", "WM"]:
            print(f"Making eroded tissue mask : {tiss}")

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
                    "singularity run",
                    "--cleanenv",
                    f"--bind {self.subj_work}:{self.subj_work}",
                    f"--bind {self.subj_work}:/opt/home",
                    self.sing_afni,
                    "3dmask_tool",
                    f"-input {bin_path}",
                    "-dilate_input -1",
                    f"-prefix {out_tiss}",
                ]
                bash_cmd = " ".join(bash_list)
                _ = submit.submit_subprocess(
                    bash_cmd, bin_path, f"Erode {tiss}"
                )

            # Add path to eroded tissue mask to attribute
            self.anat_dict[f"mask-{tiss}e"] = out_tiss

    def minimum(self):
        """Create a minimum-signal mask.

        Generate a mask for voxels in functional space that
        contain a value greater than some minimum threshold
        across all volumes and runs.

        Based around AFNI's 3dTstat -min.

        Notes
        -----
        Adds new key:value to self.anat_dict
            ["mask-minimum"] = /path/to/map/of/minimum/epi/signal

        """
        print("Making minimum value mask")

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
                    "singularity run",
                    "--cleanenv",
                    f"--bind {self.proj_deriv}:{self.proj_deriv}",
                    f"--bind {self.subj_work}:{self.subj_work}",
                    f"--bind {self.subj_work}:/opt/home",
                    self.sing_afni,
                    "3dcalc",
                    "-overwrite",
                    f"-a {run_file}",
                    "-expr 1",
                    f"-prefix {h_out_bin}",
                ]
                bash_cmd = " ".join(bash_list)
                _ = submit.submit_subprocess(bash_cmd, h_out_bin, "Binary EPI")

                # Make a mask for >min values
                h_name_min = "tmp_min_" + os.path.basename(run_file)
                h_out_min = os.path.join(self.subj_work, h_name_min)
                bash_list = [
                    "singularity run",
                    "--cleanenv",
                    f"--bind {self.subj_work}:{self.subj_work}",
                    f"--bind {self.subj_work}:/opt/home",
                    self.sing_afni,
                    "3dTstat",
                    "-min",
                    f"-prefix {h_out_min}",
                    h_out_bin,
                ]
                bash_cmd = " ".join(bash_list)
                min_list.append(
                    submit.submit_subprocess(
                        bash_cmd, h_out_min, "Minimum EPI"
                    )
                )

            # Average the minimum masks across runs
            h_name_mean = f"tmp_{self.subj}_{self.sess}_desc-mean_mask.nii.gz"
            h_out_mean = os.path.join(self.subj_work, h_name_mean)
            bash_list = [
                "singularity run",
                "--cleanenv",
                f"--bind {self.subj_work}:{self.subj_work}",
                f"--bind {self.subj_work}:/opt/home",
                self.sing_afni,
                "3dMean",
                "-datum short",
                f"-prefix {h_out_mean}",
                f"{' '.join(min_list)}",
            ]
            bash_cmd = " ".join(bash_list)
            _ = submit.submit_subprocess(bash_cmd, h_out_mean, "Mean EPI")

            # Generate mask of non-zero voxels
            bash_list = [
                "singularity run",
                "--cleanenv",
                f"--bind {self.subj_work}:{self.subj_work}",
                f"--bind {self.subj_work}:/opt/home",
                self.sing_afni,
                "3dcalc",
                f"-a {h_out_mean}",
                "-expr 'step(a-0.999)'",
                f"-prefix {out_path}",
            ]
            bash_cmd = " ".join(bash_list)
            _ = submit.submit_subprocess(bash_cmd, out_path, "MinVal EPI")

        # Update attribute with path to minimum mask
        self.anat_dict["mask-minimum"] = out_path
