"""Modeling methods for FSL-based pipelines."""
# %%
import os
import time
import glob
import json
import pandas as pd
import numpy as np
from func_model.resources.general import submit
from func_model.resources import fsl


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

    Example
    -------
    make_cf = model.ConditionFiles(**args)
    for run_num in make_cf.run_list:
        make_cf.common_events(run_num)
        make_cf.session_separate_events(run_num)

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
        if not fsl.helper.valid_task(task):
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
def confounds(
    conf_path, subj_work, na_value="n/a", fd_thresh=None, prop_thresh=0.2
):
    """Make confounds files for FSL modelling.

    Use fMRIPrep timeseries files to generate motion and censoring
    regressors. If the proportion of censored volumes is less than
    prop_thresh, a confound file will be written; failing to write
    a confound file due to excessive motion allows for
    resources.fsl.wrap.write_first_fsf to skip constructing the
    design.fsf file for the run, resulting in the run not
    being modelled in workflows.fsl_task_first.

    Confounds files are potentially written to:
        <subj_work>/confounds_files

    Censoring stats are written to:
        <subj_work>/confounds_proportions

    Parameters
    ----------
    conf_path : path
        Location of fMRIPrep confounds file
    subj_work : path
        Location of working directory for intermediates
    na_value : str, optional
        NA value in the fMRIprep confounds, will be used
        in output file.
    fd_thresh : None, float, optional
        If specified, use value to identify volumes requiring
        censoring and build output dataframe columns. Otherwise
        simply grab fMRIPrep confounds motion_outlierX columns.
    prop_thresh : float
        If the proportion of censored volumes exceeds this value,
        then the confounds file will not be written, resulting
        in first-level modelling skipping the run.

    Returns
    -------
    pd.DataFrame
        Confounds of interest data

    Raises
    ------
    FileNotFoundError
        Missing confounds file
    TypeError
        Unexpected parameter type


    """
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Expected to find file : {conf_path}")
    if not isinstance(na_value, str):
        raise TypeError("Unexpected type for na_value")
    if fd_thresh:
        if not isinstance(fd_thresh, float):
            raise TypeError("Unexpected type for fd_thresh")
    if not isinstance(prop_thresh, float):
        raise TypeError("Unexpected type for prop_thresh")

    # Setup output location
    prop_dir = os.path.join(subj_work, "confounds_proportions")
    out_dir = os.path.join(subj_work, "confounds_files")
    for _dir in [prop_dir, out_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir)

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

    # Subset dataframe for desired columns
    df = pd.read_csv(conf_path, sep="\t", na_values=na_value)
    if not fd_thresh:
        mot_cols = [x for x in df.columns if "motion_outlier" in x]
        cnt_drop = len(mot_cols) if mot_cols else 0
        col_list += mot_cols
    df_out = df[col_list].copy()

    # Add FSL motion_outlier column for each volume that matches
    # or exceeds the framewise displacement threshold
    if fd_thresh:
        mot_mask = df.index[df["framewise_displacement"] >= fd_thresh].tolist()
        cnt_drop = len(mot_mask) if mot_mask else 0
        if mot_mask:
            for cnt, idx in enumerate(mot_mask):
                df_out[f"motion_outlier{cnt:02d}"] = 0
                df_out.at[idx, f"motion_outlier{cnt:02d}"] = 1

    # Calculate, write proportion of dropped volumes
    prop_drop = round(cnt_drop / df_out.shape[0], 2) if cnt_drop != 0 else 0.0
    prop_path = os.path.join(
        prop_dir,
        os.path.basename(conf_path).replace(
            "_timeseries.tsv", "_proportion.json"
        ),
    )
    with open(prop_path, "w") as jf:
        json.dump(
            {
                "VolTotal": df_out.shape[0],
                "CensorCount": cnt_drop,
                "CensorProp": prop_drop,
            },
            jf,
        )
    if prop_drop >= prop_thresh:
        return df_out

    # Write out df
    out_name = os.path.basename(conf_path).replace(".tsv", ".txt")
    out_path = os.path.join(out_dir, out_name)
    df_out.to_csv(out_path, index=False, sep="\t", na_rep=na_value)
    return df_out


# %%
class MakeFirstFsf:
    """Generate first-level design FSF files for FSL modelling.

    Use pre-generated template FSF files to write run-specific
    first-level design FSF files for planned models.

    Design files are written to:
        <subj_work>/design_files/<run>_level-first_name-<model_name>.fsf

    Attributes
    ----------
    write_fsf
        Find appropriate method for model name, write run design.fsf

    Example
    -------
    make_fsf = model.MakeFirstFsf(**args)
    for run in [list, of, run, files]:
        fsf_path = make_fsf.write_fsf(**args)

    """

    def __init__(self, subj_work, proj_deriv, model_name):
        """Initialize.

        Read-in long and short templates.

        Parameters
        ----------
        subj_work : path
            Output work location for intermediates
        proj_deriv : path
            Location of project deriviatives directory
        model_name : str
            FSL model name, specifies template selection from
            func_model.reference_files.

        Raises
        ------
        ValueError
            Inappropriate model name or level

        """
        if not fsl.helper.valid_name(model_name):
            raise ValueError(f"Unexpected value for model_name : {model_name}")

        self._subj_work = subj_work
        self._proj_deriv = proj_deriv
        self._model_name = model_name
        self._load_templates()

    def _load_templates(self):
        """Load full, short first-level templates as private attributes."""
        self._tp_full = fsl.helper.load_reference(
            "design_template_level-first_"
            + f"name-{self._model_name}_desc-full.fsf"
        )
        self._tp_short = fsl.helper.load_reference(
            "design_template_level-first_"
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

        Update select fields of template FSF files according to user input.
        Wrapper method, identifies and executes appropriate private method
        from model_name.

        Parameters
        ----------
        run : str
            BIDS run identifier
        num_vol : int, str
            Number of EPI volumes
        preproc_path : path
            Location and name of preprocessed EPI file
        confound_path : path
            Location, name of confounds file
        judge_path : path
            Location, name of judgment condition file
        wash_path : path
            Location, name of washout condition file
        emosel_path : path
            Location, name of emotion selection condition file
        emoint_path : path
            Location, name of emotion intensity condition file
        use_short : bool
            Whether to use short or full template design

        Attributes
        ----------
        _field_switch : dict
            Find (key) and replace (value) strings for building
            run-specific design FSF from template.

        Returns
        -------
        path
            Location, name of generated design FSF

        Raises
        ------
        FileNotFoundError
            Missing input file (preproc, condition)
        TypeError
            Incorrect input type
        ValueError
            Unexpected preproc file extension

        """
        # Validate user input
        for h_path in [
            preproc_path,
            confound_path,
            judge_path,
            wash_path,
            emosel_path,
            emoint_path,
        ]:
            if not os.path.exists(h_path):
                raise FileNotFoundError(f"Missing expected file : {h_path}")
        if not isinstance(use_short, bool):
            raise TypeError("Expected use_short as type bool")

        # Set attrs, variables
        self._run = run
        self._use_short = use_short

        _pp_ext = preproc_path.split(".")[-1]
        if _pp_ext == "gz":
            pp_file = preproc_path[:-7]
        elif _pp_ext == "nii":
            pp_file = preproc_path["-4"]
        else:
            raise ValueError(
                "Expected preproc to have .nii or .nii.gz extension."
            )

        # Setup replace dictionary
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

        # Find, trigger method
        write_meth = getattr(self, f"_write_first_{self._model_name}")
        fsf_path = write_meth()
        return fsf_path

    def _stim_replay_switch(self):
        """Update replace dict for model sep.

        Find replay and stimulus emotion condition files for run,
        update private method _field_switch for model_name == sep
        specific conditions.

        Raises
        ------
        ValueError
            Unable to find replay or stimulus emotion condition file for run
            Found unequal number of replay, stimulus condition files

        """

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

        # Find stim and replay emotion condition files
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
        if len(stim_emo) != len(rep_emo):
            raise ValueError("Stimulus, replay lists unequal length")

        # Update attr
        emo_dict = _stim_replay(stim_emo, rep_emo)
        self._field_switch.update(emo_dict)

    def _write_first_sep(self):
        """Make first-level FSF for model sep.

        Write a design FSF by updating fields in the template FSF for
        model_name == sep. Write out design files to subject working
        directory.

        Returns
        -------
        path
            Location, name of design FSF file

        """

        # Update field_switch, make design file
        self._stim_replay_switch()
        fsf_edit = self._tp_short if self._use_short else self._tp_full
        for old, new in self._field_switch.items():
            fsf_edit = fsf_edit.replace(old, new)

        # Write out
        out_dir = os.path.join(self._subj_work, "design_files")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(
            out_dir,
            f"{self._run}_level-first_name-{self._model_name}_design.fsf",
        )
        with open(out_path, "w") as tf:
            tf.write(fsf_edit)
        return out_path


# %%
def run_feat(fsf_path, subj, sess, model_name, model_level, log_dir):
    """Execute FSL's feat.

    Parameters
    ----------
    fsf_path : path
        Location and name of FSL design.fsf
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    model_name : str
        FSL model name
    model_level : str
        FSL model level
    log_dir : path
        Output location for log files

    Returns
    -------
    path
        Location of output report.html

    Raises
    ------
    FileNotFoundError
        Missing report.html
    ValueError
        Inappropriate model name or level

    """
    # Validate model_name/level
    if not fsl.helper.valid_name(model_name):
        raise ValueError(f"Unexpected value for model_name : {model_name}")
    if not fsl.helper.valid_level(model_level):
        raise ValueError(f"Unexpected value for model_level: {model_level}")

    # Setup, avoid repeating work
    fsf_file = os.path.basename(fsf_path)
    run = fsf_file.split("_")[0]
    out_dir = os.path.dirname(os.path.dirname(fsf_path))
    out_path = os.path.join(
        out_dir,
        f"{run}_level-{model_level}_name-{model_name}.feat",
        "report.html",
    )
    if os.path.exists(out_path):
        return out_path

    # Schedule feat job
    job_name = subj[-4:] + "s" + sess[-1] + "feat"
    _, _ = submit.submit_sbatch(
        f"feat {fsf_path}",
        job_name,
        log_dir,
        num_hours=4,
        num_cpus=4,
        mem_gig=8,
    )
    time.sleep(30)

    # Verify output exists
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Failed to find feat output : {out_path}")
    return out_path


# %%
