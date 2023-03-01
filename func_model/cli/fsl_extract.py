"""Extract voxel beta weights from FSL FEAT files.

Written for the local labarserv2 environment.

Mine FSL GLM files for contrasts of interest and generate a
dataframe of voxel beta-coefficients. Dataframes may be masked by
identifying coordinates in a group-level mask.

Dataframes are written for each subject in --subj-list/all, and
a group dataframe can be generated from all subject dataframes.

Subject-level dataframes are titled
    <subj>_<sess>_<task>_name-<model_name>_level-<model_level>_betas.tsv
and written to:
    <proj_dir>/data_scanner_BIDS/derivatives/model_fsl/<subj>/<sess>/func

The group-level dataframe is written to:
    <proj_dir>/analyses/model_fsl/fsl_<model_name>_betas.tsv

Examples
--------
fsl_extract --sub-all
fsl_extract --sub-all --contrast-name replay
fsl_extract --sub-list sub-ER0009 sub-ER0016

"""
# %%
import os
import sys
import glob
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
    parser.add_argument(
        "--sub-list",
        nargs="+",
        help=textwrap.dedent(
            """\
            List of subject IDs to extract behavior beta-coefficients
            """
        ),
        type=str,
    )
    parser.add_argument(
        "--sub-all",
        action="store_true",
        help=textwrap.dedent(
            """\
            Extract beta-coefficients from all available subjects and
            generate a master dataframe.
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
    subj_list = args.sub_list
    subj_all = args.sub_all
    proj_dir = args.proj_dir
    con_name = args.contrast_name
    model_name = args.model_name
    model_level = args.model_level

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

    # Check, make subject list
    proj_deriv = os.path.join(
        proj_dir, "data_scanner_BIDS/derivatives/model_fsl"
    )
    subj_avail = sorted(glob.glob(f"{proj_deriv}/sub-*"))
    if not subj_avail:
        raise ValueError(f"No FSL output found at : {proj_deriv}")
    subj_avail = [os.path.basename(x) for x in subj_avail]
    if subj_list:
        for subj in subj_list:
            if subj not in subj_avail:
                raise ValueError(
                    "Specified subject not found in "
                    + f"derivatives/model_fsl : {subj}"
                )
    if subj_all:
        subj_list = subj_avail

    # Submit workflow
    workflows.fsl_extract(
        proj_dir, subj_list, model_name, model_level, con_name
    )


if __name__ == "__main__":

    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()
