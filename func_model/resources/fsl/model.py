"""Modeling methods for FSL-based pipelines.

ConditionFiles  : make condition files for task-based EPI pipelines
confounds       : make confounds files from fMRIPrep output
simul_cond_motion : identify co-ocurring events and motion for LSS
MakeFirstFsf    : write first-level design.fsf files
MakeSecondFsf   : write second-level design.fsf files
run_feat        : execute design.fsf via FEAT

"""
# %%
import os
import time
import glob
import json
import math
import pandas as pd
import numpy as np
from typing import Union, Tuple
from func_model.resources.general import submit
from func_model.resources import fsl


# %%
class ConditionFiles:
    """Make FSL-style condition files.

    Generate FSL-style condition files from BIDS task events.tsv files.
    One row for each trial, and columns for onset, duration, and modulation.

    Condition files are written to:
        <subj_work>/condition_files

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

    Methods
    -------
    common_events()
        Make condition files for common events (judgment, washout, intensity,
        selection).
    load_events(events_path)
        Read-in BIDS events file as pd.DataFrame
    session_together_events()
        For use when model_name == comb. Make condition files for
        session-specific events, combining stimulus and replay.
    session_separate_events()
        For use when model_name == sep. Make condition files for
        session-specific events, generating separate condition
        files for stimulus and replay.

    Example
    -------
    make_cond = model.ConditionFiles(*args)
    make_cond.load_events("/path/to/*events.tsv")
    comm_dict = make_cond.common_events()
    tog_dict = make_cond.session_together_events()

    """

    def __init__(self, subj, sess, task, subj_work):
        """Initialize."""
        if not fsl.helper.valid_task(task):
            raise ValueError(f"Unexpected task name : {task}")

        print("\nInitializing ConditionFiles")
        self._subj = subj
        self._sess = sess
        self._task = task
        self._subj_out = os.path.join(subj_work, "condition_files")
        if not os.path.exists(self._subj_out):
            os.makedirs(self._subj_out)

    def load_events(self, events_path):
        """Read-in BIDS events file as dataframe.

        Load events file and determine run identifier from filename.

        Parameters
        ----------
        events_path : str, os.PathLike
            Location of BIDS-formatted events file

        Raises
        ------
        FileNotFoundError
            Unable to read events_path

        """
        if not os.path.exists(events_path):
            raise FileNotFoundError(f"Expected events file : {events_path}")
        print(f"\tLoading {os.path.basename(events_path)}")
        self._df_run = pd.read_table(events_path)
        _sub, _ses, _task, self._run, _suff = os.path.basename(
            events_path
        ).split("_")

    def _write_cond(
        self, event_onset: list, event_duration: list, event_name: str
    ) -> Tuple[pd.DataFrame, os.PathLike]:
        """Compile and write conditions file."""
        if len(event_onset) != len(event_duration):
            raise ValueError(
                "Lengths of event_onset, event_duration do not match"
            )
        df = pd.DataFrame(
            {"onset": event_onset, "duration": event_duration, "mod": 1}
        )
        out_name = (
            f"{self._subj}_{self._sess}_{self._task}_{self._run}_"
            + f"desc-{event_name}_events.txt"
        )
        out_path = os.path.join(self._subj_out, out_name)
        df.to_csv(out_path, index=False, header=False, sep="\t")
        return (df, out_path)

    def session_together_events(self):
        """Generate combined stimulus+replay condition files for emotions.

        Session-specific events (scenarios, videos) are extracted and
        then condition files for each emotion are generated, with duration
        including the following replay trial.

        Returns
        -------
        dict
            key = event description
            value = path, location of condition file

        Raises
        ------
        TypeError
            run_num is not int
        ValueError
            Index and position lists are not equal

        """
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
        out_dict = {}
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
            _, emo_path = self._write_cond(
                emo_onset, emo_duration, f"tog{t_emo}"
            )
            out_dict[f"tog{t_emo}"] = emo_path
        return out_dict

    def session_separate_events(self):
        """Generate separate stimulus and replay condition files for emotions.

        Session-specific events (scenarios, videos) are extracted and
        then condition files for each emotion are generated. Separate
        condition files are made for stimulus and replay events.

        Returns
        -------
        dict
            key = event description
            value = path, location of condition file

        Raises
        ------
        TypeError
            run_num is not int
        ValueError
            Index and position lists are not equal

        """
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
        out_dict = {}
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
            _, stim_out = self._write_cond(
                stim_onset, stim_duration, f"stim{t_emo}"
            )
            _, rep_out = self._write_cond(
                replay_onset, replay_duration, f"replay{t_emo}"
            )
            out_dict[f"stim{t_emo}"] = stim_out
            out_dict[f"replay{t_emo}"] = rep_out
        return out_dict

    def session_lss_events(self):
        """Generate condition files for LSS models.

        Based on 'separate' model, first generate a set of sep condition
        files. Then, iterate through each block of stimuli and create
        trial and 'remaining' events, for each trial of each condition.

        Returns
        -------
        tuple
            [0] = dict, condition files from session_separate_events
            [1] = dict, lss condition files

        """
        # Generate typical conditions, then event-remain pairs for
        # each condition.
        cond_dict = self.session_separate_events()
        lss_dict = {}
        for cond_name, cond_path in cond_dict.items():

            # Get condition info, setup output
            lss_dict[cond_name] = {}
            cond_path = cond_dict[cond_name]
            out_dir = os.path.dirname(cond_path)
            out_pref = os.path.basename(cond_path).split("_events")[0]
            df_cond = pd.read_csv(cond_path, sep="\t", header=None)

            # Make event-remain pair for each trial
            for _idx in range(df_cond.shape[0]):
                trial_num = _idx + 1

                # Single trial
                df_trial = df_cond[df_cond.index == _idx]
                out_trial = os.path.join(
                    out_dir, f"{out_pref}_trial-{trial_num}event.txt"
                )
                df_trial.to_csv(out_trial, index=False, header=False, sep="\t")

                # Remaining trials
                df_rest = df_cond[df_cond.index != _idx]
                out_rest = os.path.join(
                    out_dir, f"{out_pref}_trial-{trial_num}remain.txt"
                )

                # Write file and update dict
                df_rest.to_csv(out_rest, index=False, header=False, sep="\t")
                lss_dict[cond_name][trial_num] = {
                    "event": out_trial,
                    "remain": out_rest,
                }
        return (cond_dict, lss_dict)

    def common_events(self):
        """Make condition files for common events of both sessions.

        Condition files for common events (judgment, washout, emotion select,
        emotion intensity) of each run are generated.

        Returns
        -------
        dict
            key = event description
            value = path, location of condition file

        Raises
        ------
        TypeError
            run_num is not int
        ValueError
            Index and position lists are not equal

        """
        # Set BIDS description field for each event
        common_switch = {
            "judge": "judgment",
            "wash": "washout",
            "emotion": "emoSelect",
            "intensity": "emoIntensity",
        }

        # Make and write condition files
        out_dict = {}
        for com, com_name in common_switch.items():
            print(f"\t\tBuilding conditions for common : {com}")
            idx_com = self._df_run.index[
                self._df_run["trial_type"] == com
            ].tolist()
            com_onset = self._df_run.loc[idx_com, "onset"].tolist()
            com_duration = self._df_run.loc[idx_com, "duration"].tolist()
            _, com_out = self._write_cond(com_onset, com_duration, com_name)
            out_dict[com_name] = com_out
        return out_dict


# %%
def confounds(conf_path, subj_work, na_value="n/a", fd_thresh=None):
    """Make confounds files for FSL modeling.

    Mine fMRIPrep timeseries.tsv files for confound regressors. Also
    calculate the proportion of volumes censored for downstream
    analytics.

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

    Returns
    -------
    tuple
        [0] = pd.DataFrame, confounds of interest data
        [1] = path, location of confound file

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
    if fd_thresh and not isinstance(fd_thresh, float):
        raise TypeError("Unexpected type for fd_thresh")

    # Setup output location
    print("\tMaking confounds")
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

    # Write out df
    out_name = os.path.basename(conf_path).replace(".tsv", ".txt")
    out_path = os.path.join(out_dir, out_name)
    df_out.to_csv(out_path, index=False, sep="\t", na_rep=na_value)
    return (df_out, out_path)


# %%
def simul_cond_motion(subj, sess, run, task, subj_work, subj_fsl):
    """Title.

    Parameters
    ----------
    subj
    sess
    run
    task
    subj_work
    subj_fsl

    """
    # Setup output location
    out_dir = os.path.join(subj_work, "condition_confounds_simultaneous")
    out_path = os.path.join(
        out_dir, f"{subj}_{sess}_{task}_{run}_desc-simul_events.txt"
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Find preproc file for TR value
    try:
        run_pp = glob.glob(f"{subj_fsl}/*{task}_{run}*_bold.nii.gz")[0]
    except IndexError:
        print(f"No preproc file for {run} found at : {subj_fsl}")
        raise
    len_tr = fsl.helper.get_tr(run_pp)

    # Find confound file
    conf_dir = os.path.join(subj_work, "confounds_files")
    try:
        conf_path = sorted(glob.glob(f"{conf_dir}/*{run}*_timeseries.txt"))[0]
    except IndexError:
        print(f"No confound file for {run} found at : {conf_dir}")
        raise

    # Load confound as dataframe, get columns of interest and determine
    # volume/scan time in seconds
    df_conf = pd.read_csv(conf_path, sep="\t")
    mot_list = [x for x in df_conf.columns if "motion_outlier" in x]
    if not mot_list:
        return
    df_mot = df_conf[mot_list].copy()
    df_mot["sum"] = df_mot[mot_list].sum(axis=1)
    df_mot["scan_time"] = df_mot.index * len_tr
    del df_conf

    # Find LSS condition files
    cond_dir = os.path.join(subj_work, "condition_files")
    cond_list = sorted(glob.glob(f"{cond_dir}/*{run}*trial*event.txt"))
    try:
        cond_list[0]
    except IndexError:
        print(f"No condition file for {run} found at : {cond_dir}")
        raise

    # Load condition file, determine event on/off times
    df_out = pd.DataFrame(columns=["stim", "trial", "count"])
    for cond_path in cond_list:
        df_cond = pd.read_csv(
            cond_path, sep="\t", names=["onset", "dur", "pm"], header=None
        )
        df_cond["offset"] = df_cond["onset"] + df_cond["dur"]
        cond_on = math.floor(df_cond.loc[0]["onset"])
        cond_off = math.ceil(df_cond.loc[0]["offset"])

        # Check if motion occurred during event
        idx_mot = df_mot.index[
            (df_mot["scan_time"] >= cond_on) & (df_mot["scan_time"] < cond_off)
        ]
        num_mot = df_mot.iloc[idx_mot]["sum"].sum(axis=0)
        if not num_mot:
            continue

        # Update df for co-occurance
        _subj, _sess, _task, _run, desc, trial = os.path.basename(
            cond_path
        ).split("_")
        df_out.loc[len(df_out)] = {
            "stim": desc.split("-")[-1],
            "trial": trial.split("-")[-1][0],
            "count": int(num_mot),
        }

    # Write out if co-occurance exists
    if df_out.shape[0]:
        df_out.to_csv(out_path, index=False, sep="\t")


# %%
def _write_design(
    out_dir: Union[str, os.PathLike], out_name: str, design_info: str
) -> Union[str, os.PathLike]:
    """Write design.fsf and return file location."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w") as tf:
        tf.write(design_info)
    return out_path


# %%
class _FirstSep:
    """Support writing first-level sep model design.fsf.

    Intended for inheritance by MakeFirstFsf, references attrs
    set by child.

    Methods
    -------
    write_sep()
        Coordinating writing of design.fsf, returns path to file

    """

    def write_sep(self):
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
        self._sep_switch()
        fsf_edit = self._tp_short if self._use_short else self._tp_full
        for old, new in self._field_switch.items():
            fsf_edit = fsf_edit.replace(old, new)

        # Write out
        # design_path = self._write_first(fsf_edit)
        out_dir = os.path.join(self._subj_work, "design_files")
        out_name = (
            f"{self._run}_level-first_name-" + f"{self._model_name}_design.fsf"
        )
        out_path = _write_design(out_dir, out_name, fsf_edit)
        return out_path

    def _sep_switch(self):
        """Update switch dictionary for model "sep".

        Find replay and stimulus emotion condition files for run,
        update private attr _field_switch for sep specific conditions.

        """

        def _get_desc(file_name: str) -> str:
            """Return description field from condition filename."""
            try:
                _su, _se, _ta, _ru, desc, _su = file_name.split("_")
                return desc.split("-")[-1]
            except IndexError:
                raise ValueError(
                    "Improperly formatted file name for condition file."
                )

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
        # TODO receive these via workflows.FslFirst._sep_cond
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


# %%
class _FirstTog:
    """Support writing first-level tog model design.fsf.

    Intended for inheritance by MakeFirstFsf, references attrs
    set by child.

    Methods
    -------
    write_tog()
        Coordinating writing of design.fsf, returns path to file

    """

    def write_tog(self):
        """Make first-level FSF for model sep.

        Write a design FSF by updating fields in the template FSF for
        model_name == tog. Write out design files to subject working
        directory.

        Returns
        -------
        path
            Location, name of design FSF file

        """
        # Update field_switch, make design file
        self._tog_switch()
        fsf_edit = self._tp_short if self._use_short else self._tp_full
        for old, new in self._field_switch.items():
            fsf_edit = fsf_edit.replace(old, new)

        # Write out
        out_dir = os.path.join(self._subj_work, "design_files")
        out_name = (
            f"{self._run}_level-first_name-" + f"{self._model_name}_design.fsf"
        )
        out_path = _write_design(out_dir, out_name, fsf_edit)
        return out_path

    def _tog_switch(self):
        """Update switch dictionary for model "tog".

        Find combined emotion condition files for run, update private attr
        _field_switch for tog specific conditions.

        """

        def _get_desc(file_name: str) -> str:
            """Return description field from condition filename."""
            try:
                _su, _se, _ta, _ru, desc, _su = file_name.split("_")
                return desc.split("-")[-1]
            except IndexError:
                raise ValueError(
                    "Improperly formatted file name for condition file."
                )

        def _tog_events(tog_emo: list) -> dict:
            """Return replacement dict for stim, replay events."""
            stim_dict = {}
            replay_dict = {}
            cnt = 1
            for tog_path in tog_emo:
                desc_stim = _get_desc(os.path.basename(tog_path))
                stim_dict[f"[[stim_emo{cnt}_name]]"] = desc_stim
                stim_dict[f"[[stim_emo{cnt}_path]]"] = tog_path
                cnt += 1
            stim_dict.update(replay_dict)
            return stim_dict

        # Find tog emotion condition files
        # TODO receive these via workflows.FslFirst._sep_cond
        tog_emo = sorted(
            glob.glob(
                f"{self._subj_work}/condition_files/*{self._run}_"
                + "desc-tog*_events.txt"
            )
        )
        if not tog_emo:
            raise ValueError("Failed to find togEmo events.")

        # Update attr
        emo_dict = _tog_events(tog_emo)
        self._field_switch.update(emo_dict)


# %%
class _FirstLss:
    """Support writing first-level LSS model designs.

    Intended for inheritance by MakeFirstFsf, references attrs
    set by child.

    Methods
    -------
    write_lss()
        Coordinating writing of design.fsf, returns list of lists

    """

    def write_lss(self) -> list:
        """Coordinate writing of all design files for each condition."""
        # Generate a set of design lists for each condition (e.g. stimRomance)
        design_list = []
        for self._stim_name, _ in self._sep_cond.items():

            # Make a switch for all names, paths of conditions that
            # are not the current iteration.
            self._switch_sep = {}
            sep_dict = {
                i: j for i, j in self._sep_cond.items() if i != self._stim_name
            }
            for _cnt, _name in enumerate(sep_dict):
                match_cnt = _cnt + 2
                _path = sep_dict[_name]
                self._switch_sep[f"[[stim_{match_cnt}_name]]"] = _name
                self._switch_sep[f"[[stim_{match_cnt}_path]]"] = _path

            # Trigger generation of all design.fsf for current condition
            design_list.append(
                self._lss_switch(self._lss_cond[self._stim_name])
            )
        return design_list

    def _lss_switch(self, lss_dict) -> list:
        """Make an lss switch for each event, trigger design generation."""
        lss_list = []
        for _num, event_dict in lss_dict.items():

            # Restart attr, set output dir name
            self._switch_lss = {}
            self._switch_lss[
                "[[bids_desc_trial]]"
            ] = f"desc-{self._stim_name}_trial-{_num}"

            # Supply name, path to single/remaining events
            self._switch_lss[
                "[[desc_trial_event_name]]"
            ] = f"{self._stim_name}_event_{_num}"
            self._switch_lss["[[desc_trial_event_path]]"] = event_dict["event"]

            self._switch_lss[
                "[[desc_trial_remain_name]]"
            ] = f"{self._stim_name}_remain_{_num}"
            self._switch_lss["[[desc_trial_remain_path]]"] = event_dict[
                "remain"
            ]

            # Generate design file
            lss_list.append(self._lss_design())
        return lss_list

    def _lss_design(self) -> Union[str, os.PathLike]:
        """Write, return path to LSS design file."""
        # Aggregate all switches
        field_switch = self._field_switch.copy()
        field_switch.update(self._switch_sep)
        field_switch.update(self._switch_lss)

        # Update fields in relevant template
        fsf_edit = self._tp_short if self._use_short else self._tp_full
        for old, new in field_switch.items():
            fsf_edit = fsf_edit.replace(old, new)

        # Write out
        out_dir = os.path.join(self._subj_work, "design_files")
        out_name = (
            f"{self._run}_level-first_name-{self._model_name}_"
            + f"{self._switch_lss['[[bids_desc_trial]]']}_design.fsf"
        )
        out_path = _write_design(out_dir, out_name, fsf_edit)
        return out_path


# %%
class MakeFirstFsf(_FirstSep, _FirstTog, _FirstLss):
    """Generate first-level design FSF files for FSL modelling.

    Inherits _FirstSep, _FirstTog, _FirstLss.

    Use pre-generated template FSF files to write run-specific
    first-level design FSF files for planned models. Design files
    are written to <subj_work>/design_files.

    Parameters
    ----------
    subj_work : str, os.PathLike
        Output work location for intermediates
    proj_deriv : str, os.PathLike
        Location of project deriviatives directory
    model_name : str
        FSL model name, specifies template selection from
        func_model.reference_files.

    Methods
    -------
    write_rest_fsf(*args)
        Generate design.fsf file for resting state EPI data
    write_task_fsf(*args, **kwargs)
        Generate design.fsf files for task-based EPI data, according to
        model_name (triggers model_name-specific private method).

    Example
    -------
    make_fsf = model.MakeFirstFsf(*args)
    task_design = make_fsf.write_task_fsf(*args, **kwargs)
    rest_design = make_fsf.write_rest_fsf(*args)

    """

    def __init__(self, subj_work, proj_deriv, model_name):
        """Initialize."""
        if not fsl.helper.valid_name(model_name):
            raise ValueError(f"Unexpected value for model_name : {model_name}")

        print("\t\tInitializing MakeFirstFSF")
        self._subj_work = subj_work
        self._proj_deriv = proj_deriv
        self._model_name = model_name
        self._load_templates()

    def _load_templates(self):
        """Load design templates."""
        if self._model_name == "rest":
            self._tp_full = fsl.helper.load_reference(
                "design_template_level-first_" + f"name-{self._model_name}.fsf"
            )
        else:
            self._tp_full = fsl.helper.load_reference(
                "design_template_level-first_"
                + f"name-{self._model_name}_desc-full.fsf"
            )
            self._tp_short = fsl.helper.load_reference(
                "design_template_level-first_"
                + f"name-{self._model_name}_desc-short.fsf"
            )

    def write_rest_fsf(
        self,
        run,
        preproc_path,
        confound_path,
    ):
        """Write first-level FSF design for resting-state EPI.

        Update select fields of template FSF files according to user input.

        Parameters
        ----------
        run : str
            BIDS run identifier
        preproc_path : str, os.PathLike
            Location of preprocessed EPI file
        confound_path : str, os.PathLike
            Location of confounds file

        Returns
        -------
        str, os.PathLike
            Location, name of generated design FSF

        """
        # Validate user input
        for h_path in [
            preproc_path,
            confound_path,
        ]:
            if not os.path.exists(h_path):
                raise FileNotFoundError(f"Missing expected file : {h_path}")
        if len(run) != 6:
            raise ValueError("Improperly formatted run description")

        # Set attrs, variables
        print("\t\t\tBuilding resting design.fsf")
        self._run = run
        pp_file = self._pp_path(preproc_path)
        num_vol = fsl.helper.count_vol(preproc_path)
        len_tr = fsl.helper.get_tr(preproc_path)

        # Setup replace dictionary, update design template
        self._field_switch = {
            "[[run]]": run,
            "[[num_vol]]": str(num_vol),
            "[[len_tr]]": str(len_tr),
            "[[preproc_path]]": pp_file,
            "[[conf_path]]": confound_path,
            "[[subj_work]]": self._subj_work,
            "[[deriv_dir]]": self._proj_deriv,
        }
        fsf_edit = self._tp_full
        for old, new in self._field_switch.items():
            fsf_edit = fsf_edit.replace(old, new)

        # Write out
        out_dir = os.path.join(self._subj_work, "design_files")
        out_name = (
            f"{self._run}_level-first_name-{self._model_name}_design.fsf"
        )
        design_path = _write_design(out_dir, out_name, fsf_edit)
        return design_path

    def write_task_fsf(
        self,
        run,
        preproc_path,
        confound_path,
        common_cond,
        use_short,
        **kwargs,
    ):
        """Write first-level FSF design for task EPI.

        Update select fields of template FSF files according to user input.
        Wrapper method, identifies and executes appropriate private method
        from model_name.

        Parameters
        ----------
        run : str
            BIDS run identifier
        preproc_path : path
            Location and name of preprocessed EPI file
        confound_path : path
            Location, name of confounds file
        common_cond : dict
            Contains paths to condition files common
            between both sessions. Requires the following keys:
            -   ["judgment"]
            -   ["washout"]
            -   ["emoSelect"]
            -   ["emoIntensity"]
        use_short : bool
            Whether to use short or full template design
        **kwargs: dict, optional
            Keyword args for LSS models: sep_cond and lss_cond

        Returns
        -------
        path, list
            Location, name of generated design FSF

        """
        # Validate user input
        for h_path in [
            preproc_path,
            confound_path,
        ]:
            if not os.path.exists(h_path):
                raise FileNotFoundError(f"Missing expected file : {h_path}")
        if not isinstance(use_short, bool):
            raise TypeError("Expected use_short as type bool")
        for req_key in ["judgment", "washout", "emoSelect", "emoIntensity"]:
            if req_key not in common_cond.keys():
                raise KeyError(
                    f"Missing expected key in common_cond : {req_key}"
                )

        # Capture LSS kwargs
        if "sep_cond" in kwargs:
            self._sep_cond = kwargs["sep_cond"]
        if "lss_cond" in kwargs:
            self._lss_cond = kwargs["lss_cond"]

        # Set helper attrs
        print("\tBuilding task design.fsf")
        self._run = run
        self._use_short = use_short

        # Start replace switch
        self._field_switch = {
            "[[run]]": run,
            "[[num_vol]]": str(fsl.helper.count_vol(preproc_path)),
            "[[preproc_path]]": self._pp_path(preproc_path),
            "[[conf_path]]": confound_path,
            "[[judge_path]]": common_cond["judgment"],
            "[[wash_path]]": common_cond["washout"],
            "[[emosel_path]]": common_cond["emoSelect"],
            "[[emoint_path]]": common_cond["emoIntensity"],
            "[[subj_work]]": self._subj_work,
            "[[deriv_dir]]": self._proj_deriv,
        }

        # Trigger model method
        write_meth = getattr(self, f"write_{self._model_name}")
        fsf_path = write_meth()
        return fsf_path

    def _pp_path(
        self, preproc_path: Union[str, os.PathLike]
    ) -> Union[str, os.PathLike]:
        """Return path to preprocessed file sans extension."""
        _pp_ext = preproc_path.split(".")[-1]
        if _pp_ext == "gz":
            return preproc_path[:-7]
        elif _pp_ext == "nii":
            return preproc_path[:-4]
        else:
            raise ValueError(
                "Expected preproc to have .nii or .nii.gz extension."
            )


# %%
class MakeSecondFsf:
    """Generate second-level design FSF files for FSL modelling.

    Use pre-generated template FSF files to write session-specific
    second-level design FSF files for planned models. Design files
    are written to <subj_work>/design_files.

    Parameters
    ----------
    subj_work : str, os.PathLike
        Output work location for intermediates
    proj_deriv : str, os.PathLike
        Location of project deriviatives directory
    model_name : str
        FSL model name, specifies template selection from
        func_model.reference_files.

    Methods
    -------
    write_task_fsf()
        Generate design.fsf files for task-based EPI data, according to
        model_name (triggers model_name-specific methods).

    Example
    -------
    make_fsf = model.MakeSecondFsf(*args)
    task_design = make_fsf.write_task_fsf()

    """

    def __init__(self, subj_work, subj_deriv, model_name):
        """Initialize."""
        print("\t\tInitializing MakeSecondFSF")
        self._subj_work = subj_work
        self._subj_deriv = subj_deriv
        self._model_name = model_name

    def write_task_fsf(self):
        """Write second-level FSF design for task EPI.

        Generate a second-level design.fsf by replacing planned
        fields in a design template.

        Returns
        -------
        str, os.PathLike
            Path to generated design.fsf

        """
        # Start switch
        field_switch = {
            "[[subj_work]]": self._subj_work,
        }

        # Find all copes, update field_switch for emotion name
        # and cope path.
        cope_dict = self._get_copes()
        cnt_cope = 1
        for cnt_ev, ev_name in enumerate(cope_dict):
            field_switch[f"[[ev_{cnt_ev + 1}_name]]"] = ev_name
            for _, cope_path in cope_dict[ev_name].items():
                field_switch[f"[[ev_{cnt_cope}_cope]]"] = cope_path
                cnt_cope += 1

        # Load template and update planned values
        design_tpl = fsl.helper.load_reference(
            f"design_template_level-second_name-{self._model_name}.fsf"
        )
        for old, new in field_switch.items():
            design_tpl = design_tpl.replace(old, new)

        # Write design file, return location
        out_dir = os.path.join(self._subj_work, "design_files")
        out_name = f"level-second_name-{self._model_name}_design.fsf"
        out_path = _write_design(out_dir, out_name, design_tpl)
        return out_path

    def _get_copes(self) -> dict:
        """Match copes to EVs of interest.

        Account for random block orders in runs, return a dict
        alphabetized by stim|replay|tog x emotion in format
        {'stimAweGTw': {1: '/*/cope?.nii.gz', 2: '/*/cope?.nii.gz'}}.

        """
        # get alphabetized copes in stimAmuse, replayAmuse order
        emo_list = [
            "Amusement",
            "Anger",
            "Anxiety",
            "Awe",
            "Calmness",
            "Craving",
            "Disgust",
            "Excitement",
            "Fear",
            "Horror",
            "Joy",
            "Neutral",
            "Romance",
            "Sadness",
            "Surprise",
        ]
        trial_list = (
            ["stim", "replay"] if self._model_name == "sep" else ["tog"]
        )

        # Mine design.con to match EVs with copes
        run_list = sorted(
            glob.glob(f"{self._subj_deriv}/run-*_name-{self._model_name}.feat")
        )
        self._run_dict = {}
        for run_path in run_list:
            run = os.path.basename(run_path).split("_")[0]
            self._run_dict[run] = self._align_con_cope(
                f"{run_path}/design.con"
            )

        # Find paths to first and second instance of
        # EV of interest (e.g. stimAmusementGTw).
        cope_dict = {}
        for emo in emo_list:
            for trial in trial_list:
                cope_dict[f"{trial}{emo}GTw"] = self._find_cope_run(
                    f"{trial}{emo}GTw"
                )
        return cope_dict

    def _align_con_cope(self, con_path: Union[str, os.PathLike]) -> dict:
        """Return contrast-cope mapping {'stimAweGTw': '/*/cope?.nii.gz'}."""
        # Get contrast lines
        con_lines = []
        with open(con_path) as cf:
            for ln in cf:
                if ln.startswith("/ContrastName"):
                    con_lines.append(ln[1:])

        # Unpack contrast lines
        cope_dir = os.path.join(os.path.dirname(con_path), "stats")
        con_dict = {}
        for ln in con_lines:
            con, name = ln.split()
            name_clean = name.split("GT")[0] + "GTw"
            con_dict[name_clean] = os.path.join(
                cope_dir, f"cope{con[-1]}.nii.gz"
            )
        return con_dict

    def _find_cope_run(self, match_str: str) -> dict:
        """Find first and second instance of cope behavior.

        Find first and second instance of match_str in self._run_dict,
        return cope path in format
        {1: "/*/cope?.nii.gz", 2: "/*/cope?.nii.gz"}.

        """
        out_dict = {}
        cnt = 1
        for _, con_dict in self._run_dict.items():
            for name, cope_path in con_dict.items():
                if name == match_str:
                    out_dict[cnt] = cope_path
                    cnt += 1
        return out_dict


# %%
def run_feat(fsf_path, subj, sess, model_name, log_dir, model_level="first"):
    """FSL feat execute design file as a scheduled child job.

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
    log_dir : path
        Output location for log files
    model_level : str, optional
        FSL model level

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
    if not fsl.helper.valid_name(model_name):
        raise ValueError(f"Unexpected value for model_name : {model_name}")
    if not fsl.helper.valid_level(model_level):
        raise ValueError(f"Unexpected value for model_level : {model_level}")

    # Setup, avoid repeating work
    fsf_file = os.path.basename(fsf_path)
    if model_level == "first":
        run = fsf_file.split("_")[0]
        feat_dir = fsf_file.split("_design")[0] + ".feat"
    elif model_level == "second":
        run = None
        feat_dir = fsf_file.split("_design")[0] + ".gfeat"
    out_dir = os.path.dirname(os.path.dirname(fsf_path))
    out_path = os.path.join(out_dir, feat_dir, "report.html")
    if os.path.exists(out_path):
        return out_path

    # Determine job name
    _job = (
        f"{subj[-4:]}_s{sess[-1]}_r{run[-1]}"
        if run
        else f"{subj[-4:]}_s{sess[-1]}"
    )
    if model_name == "lss":
        _run, _level, _name, desc, trial, _suff = fsf_file.split("_")
        job_name = f"{_job}_{desc}_{trial}_feat"
    else:
        job_name = f"{_job}_feat"

    # Submit work
    _, _ = submit.submit_sbatch(
        f"feat {fsf_path}",
        job_name,
        log_dir,
        num_hours=4,
        num_cpus=4,
        mem_gig=8,
    )

    # Give time for wrap up, verify output exists
    if not os.path.exists(out_path):
        time.sleep(120)
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Failed to find feat output : {out_path}")
    return out_path


# %%
