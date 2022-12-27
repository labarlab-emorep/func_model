r"""Conduct AFNI-based models of EPI run files.

Utilizing output of fMRIPrep, construct needed files for deconvolution. Write
the 3dDeconvolve script, and use it to generate the matrices and 3dREMLfit
script. Execute 3dREMLfit, and save output files to group location.
A workflow is submitted for each session found in subject's fmriprep
directory.

Option --model-name is used to trigger different workflows. It is planned to
support generating different timing files and 3dDeconvolve commands based
on the value of this option, but only "--model-name univ" is currently built.

Examples
--------
model_afni -s sub-ER0009
model_afni --model-name univ -s sub-ER0009 sub-ER0016

"""
# %%
import os
import sys
import time
import glob
import textwrap
from datetime import datetime
from argparse import ArgumentParser, RawTextHelpFormatter
from func_model import submit


# %%
def _get_args():
    """Get and parse arguments."""
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="univ",
        help=textwrap.dedent(
            """\
            [univ]
            AFNI model name/type, for triggering different workflows
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--proj-dir",
        type=str,
        default="/hpc/group/labarlab/EmoRep/Exp2_Compute_Emotion/data_scanner_BIDS",
        help=textwrap.dedent(
            """\
            Path to BIDS-formatted project directory
            (default : %(default)s)
            """
        ),
    )

    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "-s",
        "--sub-list",
        nargs="+",
        help=textwrap.dedent(
            """\
            List of subject IDs to submit for pre-processing
            """
        ),
        type=str,
        required=True,
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser


# %%
def main():
    """Setup working environment."""

    # Capture CLI arguments
    args = _get_args().parse_args()
    subj_list = args.sub_list
    proj_dir = args.proj_dir
    model_name = args.model_name

    # Setup group project directory, paths
    proj_deriv = os.path.join(proj_dir, "derivatives")
    proj_rawdata = os.path.join(proj_dir, "rawdata")

    # Get environmental vars
    sing_afni = os.environ["SING_AFNI"]
    user_name = os.environ["USER"]

    # Setup work directory, for intermediates
    work_deriv = os.path.join("/work", user_name, "EmoRep")
    now_time = datetime.now()
    log_dir = os.path.join(
        work_deriv,
        f"logs/func_model-afni_{now_time.strftime('%Y-%m-%d_%H:%M')}",
    )
    # log_dir = os.path.join(work_deriv, "logs/func_model-afni_test")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Submit jobs for each participant, session
    for subj in subj_list:
        sess_list = [
            os.path.basename(x)
            for x in glob.glob(
                f"{proj_deriv}/pre_processing/fmriprep/{subj}/ses-*"
            )
        ]
        if not sess_list:
            print(f"No pre-processed sessions detected for {subj}, skipping")
            continue

        for sess in sess_list:
            _, _ = submit.schedule_afni(
                subj,
                sess,
                proj_rawdata,
                proj_deriv,
                work_deriv,
                sing_afni,
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
