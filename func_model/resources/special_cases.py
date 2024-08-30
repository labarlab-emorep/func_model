"""Methods for resolving issues with subject data.

SpecCases : Connect subject IDs with solutions for idiosyncratic data issues

"""

import os
from typing import Union
import pandas as pd
from func_model.resources import helper


class SpecCases:
    """Manage special cases.

    Methods
    -------
    spec_subj()
        Determine if subject, session require special treatment
    run_spec()
        Entrypoint, provides access to adjustment methods

    Example
    -------
    spec_case = special_cases.SpecCases("sub-ER1006", "ses-day3")
    if spec_case.spec_subj():
        cond_dict = spec_case.run_spec("adjust_events", epi_path, cond_dict)
        use_short = spec_case.run_spec("adjust_short", "run-06", False)

    """

    def __init__(self, subj: str, sess: str):
        """Initialize SpecCases."""
        self._subj = subj
        self._sess = sess

    def spec_subj(self) -> bool:
        """Check if subject requires special treatment."""
        return (
            self._subj in self._spec_tx.keys()
            and self._sess in self._spec_tx[self._subj].keys()
        )

    @property
    def _spec_tx(self) -> dict:
        """Map subject and session to special treatment methods."""
        return {"sub-ER1006": {"ses-day3": ["adjust_events", "adjust_short"]}}

    def run_spec(self, step, *args):
        """Run special treatment method for step.

        Parameters
        ----------
        step : str
            {"adjust_events", "adjust_template", "adjust_short"}
        args
            Relevant input for step method

        """
        # Check for planned special treatment
        pass_args = list(args)
        if (
            not self.spec_subj()
            or step not in self._spec_tx[self._subj][self._sess]
        ):
            return pass_args

        step_meth = getattr(self, f"_{step}")
        return step_meth(*pass_args)

    def _adjust_events(
        self, epi_file: Union[str, os.PathLike], cond_dict: dict
    ) -> dict:
        """Adjust condition files for short EPI runs."""
        # Find length of run
        num_vol = helper.count_vol(epi_file)
        len_tr = helper.get_tr(epi_file)
        len_run = num_vol * len_tr

        # Remove events that occur after run ended
        for ev_name in list(cond_dict.keys()):
            ev_path = cond_dict[ev_name]
            df_cond = pd.read_csv(ev_path, sep="\t", header=None)
            df_cond = df_cond.drop(df_cond[df_cond[0] >= len_run].index)

            # Remove empty event files
            if df_cond.empty:
                os.remove(ev_path)
                del cond_dict[ev_name]
            else:
                df_cond.to_csv(ev_path, index=False, header=False, sep="\t")
        return cond_dict

    def _adjust_short(self, run: str, use_short: bool) -> bool:
        """Adjust whether to use short template."""
        tpl_map = {"sub-ER1006": {"ses-day3": {"run-06": True}}}
        if run not in tpl_map[self._subj][self._sess].keys():
            return use_short
        return tpl_map[self._subj][self._sess][run]

    def _adjust_template(self, run: str) -> str:
        """Return name of template for special case."""
        tpl_map = {
            "sub-ER1006": {
                "ses-day3": {
                    "run-06": "design_template_level-first_name-sep_"
                    + "desc-short.fsf"
                }
            }
        }
        if run not in tpl_map[self._subj][self._sess].keys():
            raise KeyError()
        return tpl_map[self._subj][self._sess][run]
