#!/bin/env /hpc/group/labarlab/research_bin/conda_envs/emorep/bin/python
"""
Run AFNI deconvolution workflow for subj, sess, model.

Example
-------
python array_wf_afni.py \\
    -e ses-day2 \\
    -m mixed \\
    -s sub-ER0009

"""

import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from func_model.workflows import wf_afni


# %%
def _get_args():
    """Get and parse arguments."""
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter
    )
    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "-e",
        dest="sess",
        choices=["ses-day2", "ses-day3"],
        type=str,
        help="BIDS Session",
        required=True,
    )
    required_args.add_argument(
        "-m",
        dest="model",
        choices=["mixed", "task", "block"],
        type=str,
        help="AFNI deconvolution name",
        required=True,
    )
    required_args.add_argument(
        "-s",
        dest="subj",
        type=str,
        help="Subject ID",
        required=True,
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    return parser


# %%
def main():
    """Conduct AFNI task workflow for subject."""
    args = _get_args().parse_args()
    sess = args.sess
    model_name = args.model
    subj = args.subj

    # Setup directories
    work_deriv = os.path.join("/work", os.environ["USER"], "EmoRep")
    log_dir = os.path.join(work_deriv, "logs", f"afni_{model_name}_batch")
    for _dir in [work_deriv, log_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir, exist_ok=True)

    # Trigger work
    wf_afni.afni_task(subj, sess, work_deriv, model_name, log_dir)


if __name__ == "__main__":
    main()
