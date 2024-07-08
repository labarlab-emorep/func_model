#!/bin/env /hpc/group/labarlab/research_bin/miniconda3/envs/dev-nate_emorep/bin/python
# TODO update interpreter for project environment
"""
Submit wf_afni.afni_task workflow for subject from scheduled array.

Notes
-----
- Intended to be submitted by array_cli.sh

Example
-------
python array_wf_afni.py -e ses-day2 -m mixed

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

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    return parser


def _subj() -> list:
    """Return list of subjects with fMRIPrep output."""
    return [
        "sub-ER0009",
        "sub-ER0016",
        "sub-ER0018",
        "sub-ER0024",
        "sub-ER0025",
        "sub-ER0031",
        "sub-ER0036",
        "sub-ER0041",
        "sub-ER0046",
        "sub-ER0052",
        "sub-ER0057",
        "sub-ER0060",
        "sub-ER0071",
        "sub-ER0072",
        "sub-ER0074",
        "sub-ER0075",
        "sub-ER0087",
        "sub-ER0093",
        "sub-ER0100",
        "sub-ER0110",
        "sub-ER0122",
        "sub-ER0126",
        "sub-ER0135",
        "sub-ER0143",
        "sub-ER0166",
        "sub-ER0181",
        "sub-ER0187",
        "sub-ER0189",
        "sub-ER0200",
        "sub-ER0208",
        "sub-ER0211",
        "sub-ER0216",
        "sub-ER0219",
        "sub-ER0221",
        "sub-ER0223",
        "sub-ER0228",
        "sub-ER0229",
        "sub-ER0234",
        "sub-ER0259",
        "sub-ER0264",
        "sub-ER0274",
        "sub-ER0309",
        "sub-ER0314",
        "sub-ER0325",
        "sub-ER0333",
        "sub-ER0349",
        "sub-ER0385",
        "sub-ER0405",
        "sub-ER0434",
        "sub-ER0443",
        "sub-ER0445",
        "sub-ER0489",
        "sub-ER0501",
        "sub-ER0511",
        "sub-ER0514",
        "sub-ER0543",
        "sub-ER0544",
        "sub-ER0552",
        "sub-ER0559",
        "sub-ER0569",
        "sub-ER0580",
        "sub-ER0581",
        "sub-ER0585",
        "sub-ER0611",
        "sub-ER0615",
        "sub-ER0630",
        "sub-ER0637",
        "sub-ER0643",
        "sub-ER0647",
        "sub-ER0679",
        "sub-ER0691",
        "sub-ER0693",
        "sub-ER0697",
        "sub-ER0712",
        "sub-ER0718",
        "sub-ER0730",
        "sub-ER0736",
        "sub-ER0740",
        "sub-ER0769",
        "sub-ER0785",
        "sub-ER0800",
        "sub-ER0802",
        "sub-ER0805",
        "sub-ER0813",
        "sub-ER0824",
        "sub-ER0899",
        "sub-ER0900",
        "sub-ER0906",
        "sub-ER0909",
        "sub-ER0911",
        "sub-ER0916",
        "sub-ER0962",
        "sub-ER0987",
        "sub-ER1002",
        "sub-ER1006",
        "sub-ER1014",
        "sub-ER1023",
        "sub-ER1058",
        "sub-ER1066",
        "sub-ER1129",
        "sub-ER1130",
        "sub-ER1131",
        "sub-ER1132",
        "sub-ER1133",
        "sub-ER1134",
        "sub-ER1135",
        "sub-ER1136",
        "sub-ER1137",
        "sub-ER1139",
        "sub-ER1152",
        "sub-ER1166",
        "sub-ER1175",
        "sub-ER1176",
        "sub-ER1177",
        "sub-ER1179",
        "sub-ER1186",
        "sub-ER1203",
        "sub-ER1211",
        "sub-ER1224",
        "sub-ER1226",
    ]


# %%
def main():
    """Use SLURM_ARRAY_TASK_ID to schedule work for subject."""
    args = _get_args().parse_args()
    sess = args.sess
    model_name = args.model

    # Identify subject
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    subj = _subj()[idx]

    # Setup directories
    work_deriv = os.path.join("/work", os.environ["USER"], "EmoRep")
    log_dir = os.path.join(work_deriv, "logs", "afni_mixed_batch")
    for _dir in [work_deriv, log_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    # Trigger work
    wf_afni.afni_task(subj, sess, work_deriv, model_name, log_dir)


if __name__ == "__main__":
    main()
