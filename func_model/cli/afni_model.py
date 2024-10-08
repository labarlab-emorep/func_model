"""Conduct AFNI-based models of EPI run files.

Written for the remote Duke Compute Cluster (DCC) environment.

Utilizing output of fMRIPrep, construct needed files for deconvolution. Write
the 3dDeconvolve script, and use it to generate the matrices and 3dREMLfit
script. Execute 3dREMLfit, and save output files to group location.
A workflow is submitted for each session found in subject's fmriprep
directory.

Model names:
    - task = Model stimulus for each emotion
    - block = Model block for each emotion
    - mixed = Model stimulus + block for each emotion
    - rest = Deprecated. Conduct a resting-state analysis referencing
        example 11 of afni_proc.py.

Requires
--------
- Global variable 'RSA_LS2' which has path to RSA key for labarserv2
- Global variable 'SING_AFNI' which has path to AFNI singularity image
- c3d executable from PATH

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
from func_model.resources import submit


# %%
def _get_args():
    """Get and parse arguments."""
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="task",
        choices=["mixed", "task", "block", "rest"],
        help=textwrap.dedent(
            """\
            AFNI model name/type, for triggering different workflows
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
            List of session BIDS IDs
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
    model_name = args.model_name

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
