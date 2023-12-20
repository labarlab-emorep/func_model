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

Examples
--------
fsl_model -s sub-ER0009 -k $RSA_LS2
fsl_model -s sub-ER0009 sub-ER0016 -k $RSA_LS2
fsl_model -s sub-ER0009 \
    -k /path/to/.ssh/id_rsa_labarserv2 \
    --model-name rest
    --preproc-type smoothed

"""

# %%
import os
import sys
import time
import socket
import textwrap
from datetime import datetime
from argparse import ArgumentParser, RawTextHelpFormatter
from func_model.resources.general import submit
from func_model.resources.fsl import helper as fsl_helper


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
        default="tog",
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

    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "-s",
        "--sub-list",
        nargs="+",
        help="List of subject IDs to submit for pre-processing",
        type=str,
        required=True,
    )
    required_args.add_argument(
        "-k",
        "--rsa-key",
        type=str,
        help="Location of labarserv2 RSA key",
        required=True,
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    return parser


# %%
def main():
    """Setup working environment."""
    args = _get_args().parse_args()
    subj_list = args.sub_list
    proj_dir = args.proj_dir
    model_name = args.model_name
    model_level = args.model_level
    rsa_key = args.rsa_key
    preproc_type = args.preproc_type

    # Check model_name, model_level
    if not fsl_helper.valid_name(model_name):
        print(f"Unsupported model name : {model_name}")
        sys.exit(1)
    if not fsl_helper.valid_level(model_level):
        print(f"Unsupported model level : {model_level}")
        sys.exit(1)
    if (
        model_name == "lss" or model_name == "rest"
    ) and model_level != "first":
        print("Second level not supported for models lss, rest")
        sys.exit(1)
    if not os.path.exists(rsa_key):
        raise FileNotFoundError(f"Expected path to RSA key, found : {rsa_key}")
    if not fsl_helper.valid_preproc(preproc_type):
        raise ValueError(f"Unspported preproc type : {preproc_type}")

    # Setup group project directory, paths
    proj_deriv = os.path.join(proj_dir, "derivatives")
    proj_rawdata = os.path.join(proj_dir, "rawdata")

    # Get environmental vars
    user_name = os.environ["USER"]
    try:
        os.environ["FSLDIR"]
    except KeyError:
        print("Missing required global variable FSLDIR")
        sys.exit(1)

    # Setup work directory, for intermediates
    work_deriv = os.path.join("/work", user_name, "EmoRep")
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
        for sess in ["ses-day2", "ses-day3"]:
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
                user_name,
                rsa_key,
            )
            time.sleep(3)


if __name__ == "__main__":
    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    host_name = socket.gethostname()
    if not env_found and "dcc" not in host_name:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()
