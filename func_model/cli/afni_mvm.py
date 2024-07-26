"""Conduct ANOVA-style testing using AFNI's 3dMVM.

MVM names:
    - amodal =

Examples
--------
afni_mvm

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
        "--emo-list",
        choices=list(helper.emo_switch().keys()),
        type=str,
        nargs="+",
        default=list(helper.emo_switch().keys()),
        help="Run tests for specified emotion(s), instead of all",
    )
    parser.add_argument(
        "--model-name",
        choices=["mixed"],
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
        "--run-mvm",
        choices=["amodal"],
        help="MVM model name",
        type=str,
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
    args = _get_args().parse_args()

    # Setup work directory, for intermediates
    work_deriv = os.path.join("/work", os.environ["USER"], "EmoRep")
    now_time = datetime.now()
    log_name = "func-afni_setup" if args.run_setup else "func-afni_mvm"
    # log_dir = os.path.join(
    #     work_deriv,
    #     "logs",
    #     f"{log_name}_{now_time.strftime('%Y-%m-%d_%H:%M')}",
    # )
    log_dir = os.path.join(work_deriv, "logs", log_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get data
    if args.run_setup:
        submit.schedule_afni_group_setup(work_deriv, log_dir)
        return

    #
    stat = args.run_mvm
    if not stat:
        return

    # # Submit workflow for each emotion
    # emo_iter = emo_list if emo_list else emo_dict.keys()
    # for emo_name in emo_iter:
    #     wf_afni.afni_mvm(proj_dir, model_name, emo_name)
    emo_list = args.emo_list
    blk_coef = args.block_coef
    model_name = args.model_name
    submit.schedule_afni_group_mvm(
        model_name, stat, emo_list, work_deriv, log_dir, blk_coef
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
