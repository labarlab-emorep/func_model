"""Conduct univariate testing using AFNI-based methods.

Written for the local labarserv2 environment.

Construct and execute simple univariate tests for sanity checking
pipeline output. Student and paired tests are organized by task
stimulus type (movies, scenarios). Output is written to:
    <proj-dir>/analyses/model_afni/ttest_<model-name>

Model names:
    - student  = Student's T-test, comparing each task emotion against zero
    - paired = Paired T-test, comparing each task emotion against washout

Examples
--------
afni_univ -s student -t movies

"""

# %%
import sys
import textwrap
import platform
from argparse import ArgumentParser, RawTextHelpFormatter
from func_model.workflows import wf_afni
from func_model.resources import helper


# %%
def _get_args():
    """Get and parse arguments."""
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--block-coef",
        action="store_true",
        help="Test block coefficients instead of event for mixed models",
    )
    parser.add_argument(
        "--model-name",
        choices=["mixed", "univ"],
        type=str,
        default="mixed",
        help=textwrap.dedent(
            """\
            AFNI deconv name
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--proj-dir",
        type=str,
        default="/mnt/keoki/experiments2/EmoRep/Exp2_Compute_Emotion",
        help=textwrap.dedent(
            """\
            Path to experiment-specific project directory
            (default : %(default)s)
            """
        ),
    )

    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "-s",
        "--stat",
        help="T-test type",
        choices=["student", "paired"],
        type=str,
        required=True,
    )
    required_args.add_argument(
        "-t",
        "--task",
        type=str,
        choices=["movies", "scenarios"],
        help="Task name",
        required=True,
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    return parser


# %%
def main():
    """Setup working environment."""
    # Validate env
    if "ccn-labarserv2" not in platform.uname().node:
        raise EnvironmentError(
            "afni_univ is written for execution on labarserv2"
        )

    args = _get_args().parse_args()
    proj_dir = args.proj_dir
    model_name = args.model_name
    stat = args.stat
    task = "task-" + args.task
    blk_coef = args.block_coef

    # Validate args
    if not helper.valid_univ_test(stat):
        raise ValueError(f"Unsupported stat name : {model_name}")
    if task not in ["task-movies", "task-scenarios"]:
        raise ValueError(f"Unexpected task name : {task}")
    if blk_coef and stat != "mixed":
        raise ValueError("--block-coef only available when model-name=mixed")

    # Run model for each task, emotion
    emo_dict = helper.emo_switch()
    for emo_name in emo_dict.keys():
        wf_afni.afni_ttest(
            task, model_name, stat, emo_name, proj_dir, blk_coef=blk_coef
        )


if __name__ == "__main__":
    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()
