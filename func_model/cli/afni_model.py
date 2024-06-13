"""Conduct AFNI-based models of EPI run files.

Written for the remote Duke Compute Cluster (DCC) environment.

Utilizing output of fMRIPrep, construct needed files for deconvolution. Write
the 3dDeconvolve script, and use it to generate the matrices and 3dREMLfit
script. Execute 3dREMLfit, and save output files to group location.
A workflow is submitted for each session found in subject's fmriprep
directory.

Model names:
    - univ = Deprecated. A standard univariate model yielding a single
        averaged beta-coefficient for each event type (-stim_times_AM1)
    - rest = Deprecated. Conduct a resting-state analysis referencing
        example 11 of afni_proc.py.
    - mixed = TODO

Output logs are written to:
    /work/$(whoami)/EmoRep/logs/func-afni_model-<model-name>_<timestamp>

Requires
--------
TODO

Examples
--------
afni_model -s sub-ER0009
afni_model \\
    -s sub-ER0009 sub-ER0016 \\
    --sess ses-day2 \\
    --model-name mixed

"""

# %%
import os
import sys
import time
import textwrap
import platform
from datetime import datetime
from argparse import ArgumentParser, RawTextHelpFormatter
from func_model.resources.general import submit
from func_model.resources.afni import helper


# %%
def _get_args():
    """Get and parse arguments."""
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mixed",
        choices=["mixed"],
        help=textwrap.dedent(
            """\
            AFNI model name/type, for triggering different workflows
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--proj-dir",
        type=str,
        default="/hpc/group/labarlab/EmoRep/Exp2_Compute_Emotion/data_scanner_BIDS",  # noqa: E501
        help=textwrap.dedent(
            """\
            Path to BIDS-formatted project directory
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--sess",
        nargs="+",
        choices=["ses-day2", "ses-day3"],
        default=["ses-day2", "ses-day3"],
        type=str,
        help=textwrap.dedent(
            """\
            "List of session BIDS IDs"
            (default : %(default)s)
            """
        ),
    )

    required_args = parser.add_argument_group("Required Arguments")

    required_args.add_argument(
        "-s",
        "--subj",
        nargs="+",
        help="List of subject BIDS IDs",
        type=str,
        required=True,
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser


# %%
def main():
    """Schedule jobs for each subject, session."""
    # Check env
    if "dcc" not in platform.uname().node:
        print("Workflow 'afni_model' is required to run on DCC.")
        sys.exit(1)

    # Capture CLI arguments
    args = _get_args().parse_args()
    subj_list = args.subj
    sess_list = args.sess
    proj_dir = args.proj_dir
    model_name = args.model_name

    #
    if model_name != "mixed":
        raise ValueError(f"Unsupported model: {model_name}")

    # Check model_name
    model_valid = helper.valid_models(model_name)
    if not model_valid:
        print(f"Unsupported model name : {model_name}")
        sys.exit(1)

    # Setup group project directory, paths
    proj_deriv = os.path.join(proj_dir, "derivatives")
    proj_rawdata = os.path.join(proj_dir, "rawdata")

    # Setup work directory, for intermediates
    work_deriv = os.path.join("/work", os.environ["USER"], "EmoRep")
    now_time = datetime.now()
    log_dir = os.path.join(
        work_deriv,
        f"logs/func-afni_model-{model_name}_"
        + f"{now_time.strftime('%Y-%m-%d_%H:%M')}",
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Submit jobs for each participant, session
    for subj in subj_list:
        for sess in sess_list:
            _, _ = submit.schedule_afni(
                subj,
                sess,
                proj_rawdata,
                proj_deriv,
                work_deriv,
                model_name,
                log_dir,
            )
            time.sleep(3)


if __name__ == "__main__":

    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()

# %%
