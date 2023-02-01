"""Modeling methods for FSL-based pipelines."""
# %%
import os
import time
import glob
import shutil
import pandas as pd
import numpy as np
import importlib.resources as pkg_resources
from func_model import reference_files
from func_model.resources.general import submit


# %%
class ConditionFiles:
    """Make FSL-style condition files for each event of each run.

    Aggregate all BIDS task events files for a participant's session,
    and then generate condition files with for each event of each run,
    using one row for each trial and columns for onset, duration, and
    modulation. Condition files are named using a BIDS format, with
    a unique value in the description field.

    Condition files are written to:
        <subj_work>/condition_files

    Attributes
    ----------
    run_list

    Methods
    -------
    session_events(run_num)
        Make condition files for session-specific events (videos, scenarios)
    common_events(run_num)
        Make condition files for common events (judgment, washout, intensity,
        selection)

    """

    def __init__(self, subj, sess, task, subj_work, sess_events):
        """Initialize.

        Parameters
        ----------
        subj : str
            BIDS subject identifier
        sess : str
            BIDS session identifier
        task : str
            BIDS task name
        subj_work : path
            Location of working directory for intermediates
        sess_events : list
            Paths to subject, session BIDS events files sorted
            by run number

        Raises
        ------
        ValueError
            Unexpected task name
            Empty sess_events

        """
        if len(sess_events) < 1:
            raise ValueError("Cannot make timing files from 0 events.tsv")
        if task not in ["task-movies", "task-scenarios"]:
            raise ValueError(f"Uncexpected task name : {task}")

        # Set attributes, make output location, make dataframe
        print("\nInitializing ConditionFiles")
        self._subj = subj
        self._sess = sess
        self._task = task
        self._sess_events = sess_events
        self._subj_cf_dir = os.path.join(subj_work, "condition_files")
        if not os.path.exists(self._subj_cf_dir):
            os.makedirs(self._subj_cf_dir)
        self._event_dataframe()

    def _event_dataframe(self):
        """Combine data from events files into dataframe.

        Attributes
        ----------
        run_list : list
            List of run integers
        _df_events : pd.DataFrame
            Column names == events files, run column added
        _events_run : list
            Run identifier extracted from event file name

        Raises
        ------
        ValueError
            The number of events files and number of runs are unequal

        """
        print("\tCompiling dataframe from session events ...")

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
        self.run_list = [int(x) for x in self._df_events["run"].unique()]

    def _get_run_df(self, run_num: int):
        """Set _df_run attribute for run data."""
        if not isinstance(run_num, int):
            raise TypeError("Expected int type for run_num")
        self._df_run = self._df_events[
            self._df_events["run"] == run_num
        ].copy()
        self._df_run = self._df_run.reset_index(drop=True)

    def _write_cond(self, event_onset, event_duration, event_name, run_num):
        """Compile and write conditions file.

        Parameters
        ----------
        event_onset : list
            Event onset times
        event_duration : list
            Event durations
        event_name : str
            Event label
        run_num : int, str
            Run label

        Returns
        -------
        pd.DataFrame
            Conditions data

        Raises
        ------
        ValueError
            event_onset, event_duration lengths unequal

        """
        if len(event_onset) != len(event_duration):
            raise ValueError(
                "Lengths of event_onset, event_duration do not match"
            )
        df = pd.DataFrame(
            {"onset": event_onset, "duration": event_duration, "mod": 1}
        )
        out_name = (
            f"{self._subj}_{self._sess}_{self._task}_run-0{run_num}_"
            + f"desc-{event_name}_events.txt"
        )
        out_path = os.path.join(self._subj_cf_dir, out_name)
        df.to_csv(out_path, index=False, header=False, sep="\t")
        return df

    def session_combined_events(self, run_num):
        """Generate combined stimulus+replay conditions for each run.

        Session-specific events (scenarios, videos) are extract for each
        run, and then condition files for each emotion are generated, with
        duration including the following replay trial.

        Condition files follow a BIDS naming scheme, with a description field
        in the format combEmotionReplay. Output files are written to:
            <subj_work>/condition_files

        Parameters
        ----------
        run_num : int
            Run number

        Raises
        ------
        TypeError
            run_num is not int
        ValueError
            Index and position lists are not equal

        """
        # Validate run_num, get data
        if not isinstance(run_num, int):
            raise TypeError("Expected int type for run_num")
        print(f"\tBuilding session conditions for run : {run_num}")
        self._get_run_df(run_num)

        # Identify indices of onset, offset, and emotions. With lists
        # being an equal length, an emotion can match in pos_emo_all
        # in order to find the onset and offset indices by following
        # the position in the lists.
        task_short = self._task.split("-")[-1]
        idx_onset = np.where(self._df_run["trial_type"] == task_short[:-1])[0]
        idx_offset = np.where(self._df_run["trial_type"] == "fix")[0]
        idx_emo_all = np.where(self._df_run["emotion"].notnull())[0]
        pos_emo_all = self._df_run.loc[idx_emo_all, "emotion"].tolist()

        # Check lists
        if len(idx_onset) != len(idx_offset):
            raise ValueError("Unequal lengths of idx_onset, idx_offset")
        if len(idx_offset) != len(pos_emo_all):
            raise ValueError("Unequal lengtsh of idx_offset, pos_emo_all")

        # Get emotion categories, clean
        emo_list = self._df_run["emotion"].unique()
        emo_list = [x for x in emo_list if x == x]
        for emo in emo_list:

            # Find onset, offset index of emotion trials
            print(f"\t\tBuilding combined conditions for emotion : {emo}")
            pos_emo = [i for i, j in enumerate(pos_emo_all) if j == emo]
            idx_emo_on = idx_onset[pos_emo]
            idx_emo_off = idx_offset[pos_emo]

            # Get onset, offset times, calculate duration. Write.
            emo_onset = self._df_run.loc[idx_emo_on, "onset"].tolist()
            emo_offset = self._df_run.loc[idx_emo_off, "onset"].tolist()
            emo_duration = [
                round(j - i, 2) for i, j in zip(emo_onset, emo_offset)
            ]
            t_emo = emo.title()
            _ = self._write_cond(
                emo_onset, emo_duration, f"comb{t_emo}Replay", run_num
            )

    def session_separate_events(self, run_num):
        """Generate separate stimulus and replay conditions for each run.

        Make condition files for each session-specific stimulus (videos,
        scenarios) and the following replay, organized by emotion and run.

        Condition files follow a BIDS naming scheme, with a description field
        in the format [stim|replay]Emotion. Output files are written to:
            <subj_work>/condition_files

        Parameters
        ----------
        run_num : int
            Run number

        Raises
        ------
        TypeError
            run_num is not int
        ValueError
            Index and position lists are not equal

        """
        # Validate run_num, get data
        if not isinstance(run_num, int):
            raise TypeError("Expected int type for run_num")
        print(f"\tBuilding session conditions for run : {run_num}")
        self._get_run_df(run_num)

        # As in session_combined_events, use list position and index to
        # align replay with the appropriate emotion.
        task_short = self._task.split("-")[-1]
        idx_stim = np.where(self._df_run["trial_type"] == task_short[:-1])[0]
        idx_replay = np.where(self._df_run["trial_type"] == "replay")[0]
        idx_emo_all = np.where(self._df_run["emotion"].notnull())[0]
        pos_emo_all = self._df_run.loc[idx_emo_all, "emotion"].tolist()

        # Get unique emotions and clean.
        emo_list = self._df_run["emotion"].unique()
        emo_list = [x for x in emo_list if x == x]
        for emo in emo_list:

            # Identify the position of the emotion in pos_emo_all
            print(f"\t\tBuilding separate conditions for emotion : {emo}")
            pos_emo = [i for i, j in enumerate(pos_emo_all) if j == emo]

            # Use emotion position to extract appropriate onset and duration
            stim_onset = self._df_run.loc[idx_stim[pos_emo], "onset"].tolist()
            stim_duration = self._df_run.loc[
                idx_stim[pos_emo], "duration"
            ].tolist()
            replay_onset = self._df_run.loc[
                idx_replay[pos_emo], "onset"
            ].tolist()
            replay_duration = self._df_run.loc[
                idx_replay[pos_emo], "duration"
            ].tolist()

            # Write condition files
            t_emo = emo.title()
            _ = self._write_cond(
                stim_onset, stim_duration, f"stim{t_emo}", run_num
            )
            _ = self._write_cond(
                replay_onset, replay_duration, f"replay{t_emo}", run_num
            )

    def common_events(self, run_num):
        """Make condition files for common events for run.

        Condition files for common events (judgment, washout, emotion select,
        emotion intensity) of each run are generated. Condition files follow
        a BIDS naming scheme, with event type in the description field.

        Output files are written to:
            <subj_work>/condition_files

        Parameters
        ----------
        run_num : int
            Run number

        Raises
        ------
        TypeError
            run_num is not int
        ValueError
            Index and position lists are not equal

        """
        # Validate run_num, get data
        if not isinstance(run_num, int):
            raise TypeError("Expected int type for run_num")
        print(f"\tBuilding common conditions for run : {run_num}")
        self._get_run_df(run_num)

        # Set BIDS description field for each event
        common_switch = {
            "judge": "judgment",
            "wash": "washout",
            "emotion": "emoSelect",
            "intensity": "emoIntensity",
        }

        # Make and write condition files
        for com, com_name in common_switch.items():
            print(f"\t\tBuilding conditions for common : {com}")
            idx_com = self._df_run.index[
                self._df_run["trial_type"] == com
            ].tolist()
            com_onset = self._df_run.loc[idx_com, "onset"].tolist()
            com_duration = self._df_run.loc[idx_com, "duration"].tolist()
            _ = self._write_cond(com_onset, com_duration, com_name, run_num)


# %%
def confounds(conf_path, subj_work, na_value="n/a"):
    """Make confounds files for FSL modelling.

    Extract relevant columns from fMRIPrep's confounds output file
    and save to a new file.

    Confounds files are written to:
        <subj_work>/confounds_files

    Parameters
    ----------
    conf_path : path
        Location of fMRIPrep confounds file
    subj_work : path
        Location of working directory for intermediates
    na_value : str, optional
        NA value in the fMRIprep confounds, will be used
        in output file.

    Returns
    -------
    pd.DataFrame
        Confounds of interest data

    Raises
    ------
    FileNotFoundError
        Missing confounds file

    """
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Expected to find file : {conf_path}")

    # Setup output location
    out_dir = os.path.join(subj_work, "confounds_files")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Specify fMRIPrep confounds of interest
    col_list = [
        "csf",
        "white_matter",
        "dvars",
        "trans_x",
        "trans_x_derivative1",
        "trans_y",
        "trans_y_derivative1",
        "trans_z",
        "trans_z_derivative1",
        "rot_x",
        "rot_x_derivative1",
        "rot_y",
        "rot_y_derivative1",
        "rot_z",
        "rot_z_derivative1",
    ]

    # Make dataframe, write out
    df = pd.read_csv(conf_path, sep="\t", na_values=na_value)
    mot_cols = [x for x in df.columns if "motion_outlier" in x]
    col_list += mot_cols
    df_out = df[col_list]

    out_name = os.path.basename(conf_path).replace(".tsv", ".txt")
    out_path = os.path.join(out_dir, out_name)
    df_out.to_csv(out_path, index=False, sep="\t", na_rep=na_value)
    return df_out


# %%
class MakeFirstFsf:
    """Title.

    Desc.

    Attributes
    ----------
    write_fsf

    """

    def __init__(self, subj_work, proj_deriv, model_name, model_level):
        """Initialize.

        Parameters
        ----------
        subj_work
        proj_deriv
        model_name
        model_level

        """
        if model_name not in ["sep"]:
            raise ValueError(f"Unexpected value for model_name : {model_name}")
        if model_level != "first":
            raise ValueError(
                f"Unexpected value for model_level : {model_level}"
            )

        self._subj_work = subj_work
        self._proj_deriv = proj_deriv
        self._model_name = model_name
        self._model_level = model_level
        self._load_templates()

    def _load_templates(self):
        """Load full and short FSF templates.

        Attributes
        ----------
        _tp_full
        _tp_short

        """

        def _load_file(file_name: str) -> str:
            """Return FSF template from resources."""
            with pkg_resources.open_text(reference_files, file_name) as tf:
                tp_line = tf.read()
            return tp_line

        self._tp_full = _load_file(
            f"design_template_level-{self._model_level}_"
            + f"name-{self._model_name}_desc-full.fsf"
        )
        self._tp_short = _load_file(
            f"design_template_level-{self._model_level}_"
            + f"name-{self._model_name}_desc-short.fsf"
        )

    def write_fsf(
        self,
        run,
        num_vol,
        preproc_path,
        confound_path,
        judge_path,
        wash_path,
        emosel_path,
        emoint_path,
        use_short,
    ):
        """Write first-level FSF design.

        Desc.

        Parameters
        ----------
        run
        num_vol
        preproc_path
        confound_path
        use_short

        Attributes
        ----------
        _field_switch

        Returns
        -------
        path

        """
        self._run = run
        self._use_short = use_short

        #
        _pp_ext = preproc_path.split(".")[-1]
        pp_file = preproc_path[:-7] if _pp_ext == "gz" else preproc_path[:-4]

        #
        self._field_switch = {
            "[[run]]": run,
            "[[num_vol]]": f"{num_vol}",
            "[[preproc_path]]": pp_file,
            "[[conf_path]]": confound_path,
            "[[judge_path]]": judge_path,
            "[[wash_path]]": wash_path,
            "[[emosel_path]]": emosel_path,
            "[[emoint_path]]": emoint_path,
            "[[subj_work]]": self._subj_work,
            "[[deriv_dir]]": self._proj_deriv,
        }

        #
        write_meth = getattr(self, f"_write_first_{self._model_name}")
        fsf_path = write_meth()
        return fsf_path

    def _stim_replay_switch(self):
        """Update _field_switch for model sep."""

        def _get_desc(file_name: str) -> str:
            """Return description field from condition filename."""
            try:
                _su, _se, _ta, _ru, desc, _su = file_name.split("_")
            except IndexError:
                raise ValueError(
                    "Improperly formatted file name for condition file."
                )
            return desc.split("-")[-1]

        def _stim_replay(stim_emo: list, rep_emo: list) -> dict:
            """Return replacement dict for stim, replay events."""
            stim_dict = {}
            replay_dict = {}
            cnt = 1
            for stim_path, rep_path in zip(stim_emo, rep_emo):
                desc_stim = _get_desc(os.path.basename(stim_path))
                stim_dict[f"[[stim_emo{cnt}_name]]"] = desc_stim
                stim_dict[f"[[stim_emo{cnt}_path]]"] = stim_path

                desc_rep = _get_desc(os.path.basename(rep_path))
                replay_dict[f"[[rep_emo{cnt}_name]]"] = desc_rep
                replay_dict[f"[[rep_emo{cnt}_path]]"] = rep_path
                cnt += 1
            stim_dict.update(replay_dict)
            return stim_dict

        #
        stim_emo = sorted(
            glob.glob(
                f"{self._subj_work}/condition_files/*{self._run}_"
                + "desc-stim*_events.txt"
            )
        )
        if not stim_emo:
            raise ValueError("Failed to find stimEmo events.")
        rep_emo = sorted(
            glob.glob(
                f"{self._subj_work}/condition_files/*{self._run}_"
                + "desc-repl*_events.txt"
            )
        )
        if not rep_emo:
            raise ValueError("Failed to find repEmo events.")

        #
        emo_dict = _stim_replay(stim_emo, rep_emo)
        self._field_switch.update(emo_dict)

    def _write_first_sep(self):
        """Make first-level FSF for model sep.

        Returns
        -------
        path

        """

        # Update field_switch
        self._stim_replay_switch()

        #
        if self._use_short:
            fsf_edit = self._tp_short
        else:
            fsf_edit = self._tp_full
        for old, new in self._field_switch.items():
            fsf_edit = fsf_edit.replace(old, new)

        #
        out_dir = os.path.join(
            self._subj_work,
            f"{self._run}_level-{self._model_level}_name-{self._model_name}",
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, "design.fsf")
        with open(out_path, "w") as tf:
            tf.write(fsf_edit)
        return out_path


# %%
def run_feat(fsf_path, subj, sess, log_dir):
    """Title."""
    # check for output file
    out_dir = os.path.dirname(fsf_path)
    out_path = os.path.join(f"{out_dir}.feat", "report.html")
    if os.path.exists(out_path):
        return

    #
    job_name = subj[-4:] + "s" + sess[-1] + "feat"
    _, _ = submit.submit_sbatch(
        f"feat {fsf_path}",
        job_name,
        log_dir,
        num_cpus=4,
        mem_gig=8,
    )
    time.sleep(30)

    #
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Failed to find feat output : {out_path}")
    shutil.rmtree(out_dir)


# %%
