"""Methods for FSL-based pipelines."""
import os
import pandas as pd


class ConditionFiles:
    """Title.

    Desc.

    """

    def __init__(self, subj_work, sess_events):
        """Title.

        Parameters
        ----------
        subj_work
        sess_events

        """
        # Check arguments
        if len(sess_events) < 1:
            raise ValueError("Cannot make timing files from 0 events.tsv")

        # Set attributes, make output location
        self.sess_events = sess_events
        self.subj_cf_dir = os.path.join(subj_work, "condition_files")
        if not os.path.exists(self.subj_cf_dir):
            os.makedirs(self.subj_cf_dir)

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
