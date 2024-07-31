r"""Conduct linear mixed effects testing via AFNI's 3dLMEr.

Test for main effects of emotions, tasks, and emotion x task interactions
treating subjects as random effects via:
    Y = emotion*task+(1|Subj)+(1|Subj:emotion)+(1|Subj:task)

Three steps are involved in setting up and executing this analysis:
downloading data from keoki, determining subbrick IDs, and the
actual LME model (see Example below).

Model names correspond to afni_model output:
    - task = Model stimulus for each emotion
    - block = Model block for each emotion
    - mixed = Model stimulus + block for each emotion

When using --model-name=mixed, the default behavior is to
extract the task/stimulus subbricks. The block subbrick is
available by including the option --block-coef.

Requires
--------
- Global variable 'RSA_LS2' which has path to RSA key for labarserv2
- Global variable 'SING_AFNI' which has path to AFNI singularity image

Notes
-----
Validated on AFNI Version: Precompiled binary linux_ubuntu_24_64: Jul 16 2024
(Version AFNI_24.2.01 'Macrinus') AFNI_24.2.01 'Macrinus', see
https://hub.docker.com/r/nmuncy/afni_ub24

Example
-------
1. Get necessary data
    afni_lmer \
        --run-setup \
        --model-name mixed

2. Identify sub-brick labels
    afni_lmer \
        --get-subbricks \
        --model-name mixed \
        --block-coef

3. Conduct LME
    afni_lmer \
        --run-lmer \
        --model-name mixed \
        --block-coef

4. Conduct Monte Carlo simulations
    afni_lmer \
        --run-mc \
        --model-name mixed

"""

# %%
import os
import sys
import textwrap
from datetime import datetime
import platform
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
        "--get-subbricks",
        action="store_true",
        help="Identify sub-brick labels for emotions",
    )
    parser.add_argument(
        "--emo-list",
        choices=list(helper.emo_switch().keys()),
        type=str,
        nargs="+",
        default=list(helper.emo_switch().keys()),
        help="Run tests for specified emotion(s), instead of all",
    )
    parser.add_argument(
        "--model-name",
        choices=["mixed", "task", "block"],
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
        "--run-mc",
        action="store_true",
        help="Conduct monte carlo simulations",
    )
    parser.add_argument(
        "--run-lmer",
        action="store_true",
        help="Conduct linear mixed effect model",
    )
    parser.add_argument(
        "--run-setup",
        action="store_true",
        help="Download model_afni data and make template mask",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    return parser


# %%
def main():
    """Capture, validate arguments and submit workflow."""
    # Validate env
    if "dcc" not in platform.uname().node:
        raise EnvironmentError("afni_mvm is written for execution on DCC")

    # Get args
    args = _get_args().parse_args()
    model_name = args.model_name
    blk_coef = args.block_coef
    get_sub = args.get_subbricks
    run_lmer = args.run_lmer
    run_setup = args.run_setup
    run_mc = args.run_mc
    emo_list = args.emo_list

    # Setup log dirs
    work_deriv = os.path.join("/work", os.environ["USER"], "EmoRep")
    if run_setup:
        log_name = "func-afni_setup"
    elif get_sub:
        log_name = "func-afni_subbricks"
    elif run_lmer:
        log_name = "func-afni_lmer"
    elif run_mc:
        log_name = "func-afni_montecarlo"
    else:
        raise ValueError()

    now_time = datetime.now()
    log_dir = os.path.join(
        work_deriv,
        "logs",
        f"{log_name}_{now_time.strftime('%Y-%m-%d_%H:%M')}",
    )
    log_dir = os.path.join(work_deriv, "logs", log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get data
    if run_setup:
        submit.schedule_afni_group_setup(
            "task-movies", model_name, work_deriv, log_dir
        )
        submit.schedule_afni_group_setup(
            "task-scenarios", model_name, work_deriv, log_dir
        )
        return

    # Run MC
    if run_mc:
        submit.schedule_afni_group_mc(model_name, work_deriv, log_dir)

    # Validate opts
    if blk_coef and model_name != "mixed":
        raise ValueError("--block-coef only available when model-name=mixed")

    # Get subbricks
    if get_sub:
        submit.schedule_afni_group_subbrick(
            "task-movies",
            model_name,
            "lmer",
            emo_list,
            work_deriv,
            log_dir,
            blk_coef,
        )
        submit.schedule_afni_group_subbrick(
            "task-scenarios",
            model_name,
            "lmer",
            emo_list,
            work_deriv,
            log_dir,
            blk_coef,
        )
        return

    # Run LMEr
    if run_lmer:
        submit.schedule_afni_group_lmer(
            model_name, emo_list, work_deriv, log_dir, blk_coef
        )


if __name__ == "__main__":
    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()

# %%
