"""Conduct univariate testing using AFNI-based methods.

Construct and execute simple univariate tests for sanity checking
pipeline output. Output is written to:
    <proj-dir>/analyses/model_afni/ttest_<model-name>

Model names:
    - student  =
    - pairwise =

Examples
--------
afni_univ -n student
afni_univ -n pairwise --task-name movies

"""
# %%
import os
import sys
import textwrap
from argparse import ArgumentParser, RawTextHelpFormatter
from func_model import workflows
from func_model.resources import afni


# %%
def _get_args():
    """Get and parse arguments."""
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter
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
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help=textwrap.dedent(
            """\
            [movies | scenarios]
            If specified, conduct model-name for only specified task.
            (default : %(default)s)
            """
        ),
    )

    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "-n",
        "--model-name",
        help=textwrap.dedent(
            """\
            [student | pairwise]
            Name of model, for organizing output and triggering
            differing workflows.
            """
        ),
        type=str,
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
    proj_dir = args.proj_dir
    model_name = args.model_name
    task_name = args.task_name

    # Check model_name
    univ_models = ["pairwise", "student"]
    if model_name not in univ_models:
        print(f"Unsupported model name : {model_name}")
        sys.exit(1)

    # Setup
    group_dir = os.path.join(proj_dir, "analyses/model_afni")
    if not os.path.exists(group_dir):
        os.makedirs(group_dir)

    emo_dict = afni.helper.emo_switch()
    task_list = ["task-movies", "task-scenarios"]
    if task_name:
        chk_task = f"task-{task_name}"
        if chk_task not in task_list:
            raise ValueError(f"Unexpected task name : {task_name}")
        task_list = [chk_task]

    # Flight control for model names
    if model_name in univ_models:
        for task in task_list:
            for emo_name in emo_dict.keys():
                workflows.afni_ttest(
                    task, model_name, emo_name, proj_dir, group_dir
                )


if __name__ == "__main__":

    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()
