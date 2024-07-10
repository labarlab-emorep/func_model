r"""Conduct T-Testing using AFNI's ETAC methods.

Model names:
    - task
    - block
    - mixed

Stat names:
    - student = Student's T-test, compare each task emotion against zero
    - paired = Paired T-test, compare each task emotion against washout

Notes
-----
- Option --block-coef only available for --model-name=mixed.

Example
-------
1. Get necessary data
    afni_etac \
        --run-setup \
        --task movies \
        --model-name mixed

2. Identify sub-brick labels
    afni_etac --get-subbricks \
        --stat paired \
        --task movies \
        --model-name mixed \
        --block-coef

3. Conduct T-testing
    afni_etac --run-etac \
        --stat paired \
        --task movies \
        --model-name mixed \
        --block-coef

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
        help="Test block (instead of event) coefficients "
        + "when model-name=mixed",
    )
    parser.add_argument(
        "--emo-name",
        choices=list(helper.emo_switch().keys()),
        type=str,
        help="Use emotion (instead of all) for workflow",
    )
    parser.add_argument(
        "--get-subbricks",
        action="store_true",
        help="Identify sub-brick labels for emotions and washout",
    )
    parser.add_argument(
        "--model-name",
        choices=["mixed", "task", "block"],
        type=str,
        default="task",
        help=textwrap.dedent(
            """\
            AFNI deconv name
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--run-etac",
        action="store_true",
        help="Conduct t-testing via AFNI's ETAC",
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
        raise EnvironmentError("afni_etac is written for execution on DCC")

    # Catch args
    args = _get_args().parse_args()
    model_name = args.model_name
    blk_coef = args.block_coef
    stat = args.stat
    task = "task-" + args.task
    emo_name = args.emo_name
    get_subs = args.get_subbricks

    # Setup work directory, for intermediates
    work_deriv = os.path.join("/work", os.environ["USER"], "EmoRepTest")
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
        submit.schedule_afni_group_setup(task, model_name, work_deriv, log_dir)
        return

    # Validate args
    if not helper.valid_univ_test(stat):
        raise ValueError(f"Unsupported stat name : {stat}")
    if task not in ["task-movies", "task-scenarios"]:
        raise ValueError(f"Unexpected task name : {task}")
    if blk_coef and model_name != "mixed":
        raise ValueError("--block-coef only available when model-name=mixed")

    # Get subbricks
    emo_list = [emo_name] if emo_name else list(helper.emo_switch().keys())
    if get_subs:
        submit.schedule_afni_group_subbrick(
            task, model_name, stat, emo_list, work_deriv, log_dir, blk_coef
        )
        return

    # Schedule model for each task, emotion
    if args.run_etac:
        for emo in emo_list:
            submit.schedule_afni_group_etac(
                task, model_name, stat, emo, work_deriv, log_dir, blk_coef
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
