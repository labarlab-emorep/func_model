"""Methods for FSL-based pipelines."""
# %%
import os
import pandas as pd
import numpy as np


# %%
class ConditionFiles:
    """Title.

    Desc.

    """

    def __init__(self, subj, sess, task, subj_work, sess_events):
        """Title.

        Parameters
        ----------
        subj_work
        sess_events

        """
        print("\nInitializing ConditionFiles")

        # TODO Check arguments
        if len(sess_events) < 1:
            raise ValueError("Cannot make timing files from 0 events.tsv")

        # Set attributes, make output location
        self.subj = subj
        self.sess = sess
        self.task = task
        self.sess_events = sess_events
        self.subj_cf_dir = os.path.join(subj_work, "condition_files")
        if not os.path.exists(self.subj_cf_dir):
            os.makedirs(self.subj_cf_dir)

        #
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
        print("\tCompiling dataframe from session events ...")
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
        self.run_list = self.df_events["run"].unique()

    def _get_run_df(self, run_num):
        """Title."""
        #
        self.df_run = self.df_events[self.df_events["run"] == run_num].copy()
        self.df_run = self.df_run.reset_index(drop=True)

    def _write_cond(self, event_onset, event_duration, event_name, run_num):
        """Title."""
        df = pd.DataFrame(
            {"onset": event_onset, "duration": event_duration, "mod": 1}
        )
        out_name = (
            f"{self.subj}_{self.sess}_task-{self.task}_run-0{run_num}_"
            + f"desc-{event_name}_events.txt"
        )
        out_path = os.path.join(self.subj_cf_dir, out_name)
        df.to_csv(out_path, index=False, header=False, sep="\t")
        return df

    def session_events(self, run_num):
        """Title.

        Parameters
        ----------
        subj
        sess
        task
        run_num

        """
        print(f"\tBuilding session conditions for run : {run_num}")

        # Validate run_num

        #
        self._get_run_df(run_num)

        #
        idx_onset = np.where(self.df_run["trial_type"] == self.task[:-1])[0]
        idx_offset = np.where(self.df_run["trial_type"] == "fix")[0]
        idx_emo_all = np.where(self.df_run["emotion"].notnull())[0]
        emo_list_all = self.df_run.loc[idx_emo_all, "emotion"].tolist()

        # TODO Validate idx lists

        #
        emo_list = self.df_run["emotion"].unique()
        emo_list = [x for x in emo_list if x == x]
        for emo in emo_list:

            #
            print(f"\t\tBuilding conditions for emotion : {emo}")
            pos_emo = [i for i, j in enumerate(emo_list_all) if j == emo]
            idx_emo_on = idx_onset[pos_emo]
            idx_emo_off = idx_offset[pos_emo]

            #
            emo_onset = self.df_run.loc[idx_emo_on, "onset"].tolist()
            emo_offset = self.df_run.loc[idx_emo_off, "onset"].tolist()
            emo_duration = [
                round(j - i, 2) for i, j in zip(emo_onset, emo_offset)
            ]
            _ = self._write_cond(emo_onset, emo_duration, emo, run_num)

    def common_events(self, run_num):
        """Title.

        Desc.

        """
        print(f"\tBuilding common conditions for run : {run_num}")

        # Validate run_num

        #
        self._get_run_df(run_num)

        common_switch = {
            "judge": "judgment",
            "wash": "washout",
            "emotion": "emoSelect",
            "intensity": "emoIntensity",
        }
        for com, com_name in common_switch.items():
            print(f"\t\tBuilding conditions for common : {com}")
            idx_com = self.df_run.index[
                self.df_run["trial_type"] == com
            ].tolist()
            com_onset = self.df_run.loc[idx_com, "onset"].tolist()
            com_duration = self.df_run.loc[idx_com, "duration"].tolist()
            _ = self._write_cond(com_onset, com_duration, com_name, run_num)


# %%
def confounds(conf_path, subj_work, na_value="n/a"):
    """Title.

    Desc.

    """
    out_dir = os.path.join(subj_work, "confounds_files")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
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
    df = pd.read_csv(conf_path, sep="\t", na_values=na_value)
    mot_cols = [x for x in df.columns if "motion_outlier" in x]
    col_list += mot_cols
    df_out = df[col_list]

    out_name = os.path.basename(conf_path).replace(".tsv", ".txt")
    out_path = os.path.join(out_dir, out_name)
    df_out.to_csv(out_path, index=False, sep="\t", na_rep=na_value)
    return df_out


# %%
