r"""CLI for initiating FSL regressions.

Written for the remote Duke Compute Cluster (DCC) environment.

Setup and run first- and second-level models in FSL for task
and resting-state EPI data. Output are written to participant
derivatives:
    <proj-dir>/derivatives/model_fsl/<subj>/<sess>/func/run-*_<level>_<name>

Model names:
    - sep = emotion stimulus (scenarios, movies) and replay are
        modeled separately
    - tog = emotion stimulus and replay modeled together
    - rest = model resting-state data to remove nuissance regressors,
        first-level only
    - lss = similar to tog, but with each trial separate,
        first-level only

Level names:
    - first = first-level GLM
    - second = second-level GLM, for model-name=sep|tog only

Notes
-----
- Requires environmental variable 'RSA_LS2' to contain
    location of RSA key for labarserv2

Examples
--------
fsl_model -s sub-ER0009
fsl_model -s sub-ER0009 sub-ER0016 \
    --model-name tog \
    --ses-list ses-day2 \
    --preproc-type smoothed

"""

# %%
import os
import sys
import time
import platform
import textwrap
from datetime import datetime
from argparse import ArgumentParser, RawTextHelpFormatter
from func_model.resources import submit
from func_model.resources import helper


# %%
def _get_args():
    """Get and parse arguments."""
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--model-level",
        type=str,
        default="first",
        choices=["first", "second"],
        help=textwrap.dedent(
            """\
            FSL model level, for triggering different workflows.
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sep",
        choices=["sep", "tog", "rest", "lss"],
        help=textwrap.dedent(
            """\
            FSL model name, for triggering different workflows
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--preproc-type",
        type=str,
        default="scaled",
        choices=["scaled", "smoothed"],
        help=textwrap.dedent(
            """\
            Determine whether to use scaled or smoothed preprocessed EPIs
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
        "--ses-list",
        nargs="+",
        choices=["ses-day2", "ses-day3"],
        default=["ses-day2", "ses-day3"],
        help=textwrap.dedent(
            """\
            List of subject IDs to submit for pre-processing
            (default : %(default)s)
            """
        ),
        type=str,
    )

    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "-s",
        "--sub-list",
        nargs="+",
        help="List of subject IDs to submit for pre-processing",
        type=str,
        required=True,
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    return parser


# %%
def main():
    """Trigger workflow."""
    # Check env
    if "dcc" not in platform.uname().node:
        print("fsl_model workflow is required to run on DCC.")
        sys.exit(1)

    # Get cli input
    args = _get_args().parse_args()
    subj_list = args.sub_list
    sess_list = args.ses_list
    proj_dir = args.proj_dir
    model_name = args.model_name
    model_level = args.model_level
    preproc_type = args.preproc_type

    # Check model_name, model_level
    if not helper.valid_name(model_name):
        print(f"Unsupported model name : {model_name}")
        sys.exit(1)
    if not helper.valid_level(model_level):
        print(f"Unsupported model level : {model_level}")
        sys.exit(1)
    if (
        model_name == "lss" or model_name == "rest"
    ) and model_level != "first":
        print("Second level not supported for models lss, rest")
        sys.exit(1)
    if not helper.valid_preproc(preproc_type):
        raise ValueError(f"Unspported preproc type : {preproc_type}")

    # Setup group project directory, paths
    proj_deriv = os.path.join(proj_dir, "derivatives")
    proj_rawdata = os.path.join(proj_dir, "rawdata")

    # Check environmental vars
    for chk_env in ["FSLDIR", "RSA_LS2"]:
        try:
            os.environ[chk_env]
        except KeyError:
            print(f"Missing required environmental variable {chk_env}")
            sys.exit(1)

    # Setup work directory, for intermediates
    work_deriv = os.path.join("/work", os.environ["USER"], "EmoRep")
    now_time = datetime.now()
    log_dir = os.path.join(
        work_deriv,
        f"logs/func-fsl_model-{model_name}_"
        + f"{now_time.strftime('%Y-%m-%d_%H:%M')}",
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Submit jobs for each participant, session
    for subj in subj_list:
        for sess in sess_list:
            _, _ = submit.schedule_fsl(
                subj,
                sess,
                model_name,
                model_level,
                preproc_type,
                proj_rawdata,
                proj_deriv,
                work_deriv,
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
