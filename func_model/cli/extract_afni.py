r"""Title

Examples
--------
extract_afni -s sub-ER0009

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
from func_model import afni


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
    parser.add_argument(
        "--sub-list",
        nargs="+",
        help=textwrap.dedent(
            """\
            List of subject IDs to extract behavior beta-coefficients
            """
        ),
        type=str,
    )
    parser.add_argument(
        "--sub-all",
        action="store_true",
        help=textwrap.dedent(
            """\
            Extract beta-coefficients from all available subjects and
            generate a master dataframe.
            """
        ),
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

    # Check model_name
    if model_name not in ["univ"]:
        print(f"Unsupported model name : {model_name}")
        sys.exit(1)

    # Setup group project directory, paths
    proj_deriv = os.path.join(proj_dir, "derivatives")
    proj_rawdata = os.path.join(proj_dir, "rawdata")

    # Get environmental vars
    sing_afni = os.environ["SING_AFNI"]
    user_name = os.environ["USER"]

    # Setup work directory, for intermediates
    work_deriv = os.path.join("/work", user_name, "EmoRep")
    now_time = datetime.now()
    # log_dir = os.path.join(
    #     work_deriv,
    #     f"logs/func-afni_model-{model_name}_"
    #     + f"{now_time.strftime('%Y-%m-%d_%H:%M')}",
    # )
    log_dir = os.path.join(work_deriv, "logs/func_model-afni_test")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # # Submit jobs for each participant, session
    # for subj in subj_list:
    #     # TODO check for deconvole output
    #     time.sleep(3)


if __name__ == "__main__":

    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()
