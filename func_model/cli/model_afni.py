r"""Title.

Examples
--------
model_afni -s sub-ER0009

"""
# %%
import os
import sys
import time
import textwrap
from datetime import datetime
from argparse import ArgumentParser, RawTextHelpFormatter
from func_model import submit


# %%fnma
def _get_args():
    """Get and parse arguments."""
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--proj-dir",
        type=str,
        default="/hpc/group/labarlab/EmoRep_BIDS",
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

    # Setup group project directory, paths
    proj_deriv = os.path.join(proj_dir, "derivatives")

    # Get environmental vars
    sing_afni = os.environ["SING_AFNI"]
    user_name = os.environ["USER"]

    # Setup work directory, for intermediates
    proj_name = os.path.basename(proj_dir)
    work_deriv = os.path.join("/work", user_name, proj_name, "derivatives")
    now_time = datetime.now()
    log_dir = os.path.join(
        work_deriv,
        f"logs/func_model-afni_{now_time.strftime('%y-%m-%d_%H:%M')}",
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Submit jobs for subj_list
    for subj in subj_list:
        _, _ = submit.schedule_afni(
            subj,
            proj_deriv,
            work_deriv,
            sing_afni,
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
