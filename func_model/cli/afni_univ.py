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
afni_univ --run-setup
afni_univ --stat student --task scenarios
afni_univ --stat paired --task movies --block-coef

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
        "--run-setup",
        action="store_true",
        help="Download model_afni data and make template mask",
    )
    parser.add_argument(
        "--stat",
        help="T-test type",
        choices=["student", "paired"],
        type=str,
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["movies", "scenarios"],
        help="Task name",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    return parser


# %%
def main():
    """Setup working environment."""
    # Validate env
    if "dcc" not in platform.uname().node:
        raise EnvironmentError("afni_univ is written for execution on DCC")

    # Catch args
    args = _get_args().parse_args()
    model_name = args.model_name
    blk_coef = args.block_coef
    stat = args.stat
    task = "task-" + args.task

    # Setup work directory, for intermediates
    work_deriv = os.path.join("/work", os.environ["USER"], "EmoRep")
    now_time = datetime.now()
    log_name = (
        "func-afni_setup"
        if args.run_setup
        else f"func-afni_stat-{stat}_{task}_model-{model_name}"
    )
    log_dir = os.path.join(
        work_deriv,
        "logs",
        f"{log_name}_{now_time.strftime('%Y-%m-%d_%H:%M')}",
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get data
    if args.run_setup:
        submit.schedule_afni_group_setup(work_deriv, log_dir)
        return

    # Validate args
    if not helper.valid_univ_test(stat):
        raise ValueError(f"Unsupported stat name : {model_name}")
    if task not in ["task-movies", "task-scenarios"]:
        raise ValueError(f"Unexpected task name : {task}")
    if blk_coef and model_name != "mixed":
        raise ValueError("--block-coef only available when model-name=mixed")

    # Schedule model for each task, emotion
    emo_dict = helper.emo_switch()
    for emo_name in emo_dict.keys():
        submit.schedule_afni_group_univ(
            task, model_name, stat, emo_name, work_deriv, log_dir, blk_coef
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
