r"""Generate NIfTI masks from classifier output.

Written for the local labarserv2 environment.

Convert data from db_emorep.tbl_plsda_binary_gm into
NIfTI files build in MNI template space. Then generate
conjunctive analysis maps.

Writes output to:
    proj_dir/analyses/classify_fMRI_plsda/voxel_importance_maps/name-*_task-*_maps

Examples
--------
fsl_map -t movies
fsl_map -t both \
    --contrast-name tog \
    --model-name tog

"""

# %%
import os
import sys
import textwrap
import platform
from argparse import ArgumentParser, RawTextHelpFormatter
from func_model.workflows import wf_fsl
from func_model.resources.fsl import helper as fsl_helper


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
        choices=["stim", "replay", "tog"],
        help=textwrap.dedent(
            """\
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
        choices=["first"],
        help=textwrap.dedent(
            """\
            FSL model level, for triggering different workflows
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sep",
        choices=["sep", "tog"],
        help=textwrap.dedent(
            """\
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
        choices=["movies", "scenarios", "both"],
        required=True,
        help=textwrap.dedent(
            """\
            Name of EmoRep stimulus type, corresponds to BIDS task field.
            Used to identify data used for classifier, 'both' = classifier
            trained on movies + scenarios data.
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
    """Trigger workflow."""
    # Check env
    if "labarserv2" not in platform.uname().node:
        print("fsl_group is required to run on labarserv2.")
        sys.exit(1)

    # Get CLI input
    args = _get_args().parse_args()
    proj_dir = args.proj_dir
    con_name = args.contrast_name
    model_name = args.model_name
    model_level = args.model_level
    task_name = args.task_name

    # Check user input
    if not fsl_helper.valid_level(model_level):
        print(f"Unsupported model level : {model_level}")
        sys.exit(1)
    if not fsl_helper.valid_contrast(con_name):
        print(f"Unsupported contrast name : {con_name}")
        sys.exit(1)

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
        raise FileNotFoundError(f"Expected to find template : {tpl_path}")

    # Submit workflow
    wf_fsl.fsl_classify_mask(
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
