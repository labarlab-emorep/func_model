"""Title.

Written for the local labarserv2 environment.



Examples
--------
fsl_map -t movies

"""
# %%
import os
import sys
import textwrap
from argparse import ArgumentParser, RawTextHelpFormatter
from func_model import workflows
from func_model.resources import fsl


# %%
def _get_args():
    """Get and parse arguments."""
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--contrast-name",
        type=str,
        default="stim",
        help=textwrap.dedent(
            """\
            [stim | replay]
            Desired contrast from which coefficients will be extracted,
            substring of design.fsf EV Title.
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--model-level",
        type=str,
        default="first",
        help=textwrap.dedent(
            """\
            [first]
            FSL model level, for triggering different workflows
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sep",
        help=textwrap.dedent(
            """\
            [sep]
            FSL model name, for triggering different workflows
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
        "-t",
        "--task-name",
        type=str,
        required=True,
        help=textwrap.dedent(
            """\
            [movies | scenarios]
            Name of EmoRep stimulus type, corresponds to BIDS task field
            (default : %(default)s)
            """
        ),
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
    con_name = args.contrast_name
    model_name = args.model_name
    model_level = args.model_level
    task_name = args.task_name

    # Check user input
    if not fsl.helper.valid_name(model_name):
        print(f"Unsupported model name : {model_name}")
        sys.exit(1)
    if not fsl.helper.valid_level(model_level):
        print(f"Unsupported model level : {model_level}")
        sys.exit(1)
    if not fsl.helper.valid_contrast(con_name):
        print(f"Unsupported contrast name : {con_name}")
        sys.exit(1)
    if task_name not in ["movies", "scenarios"]:
        raise ValueError(f"Unexpected value for task : {task_name}")

    # Get template path
    try:
        tplflow_dir = os.environ["SINGULARITYENV_TEMPLATEFLOW_HOME"]
    except KeyError:
        raise EnvironmentError(
            "Expected global variable SINGULARITYENV_TEMPLATEFLOW_HOME"
            + "try : $labar_env emorep"
        )
    tpl_path = os.path.join(
        tplflow_dir,
        "tpl-MNI152NLin6Asym",
        "tpl-MNI152NLin6Asym_res-02_T1w.nii.gz",
    )
    if not os.path.exists(tpl_path):
        raise (FileNotFoundError(f"Expected to find template : {tpl_path}"))

    # Submit workflow
    workflows.fsl_classify_mask(
        proj_dir, model_name, model_level, con_name, task_name, tpl_path
    )


if __name__ == "__main__":

    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()
