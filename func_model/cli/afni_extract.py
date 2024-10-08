"""Extract voxel beta weights from deconvolved files.

Written for the local labarserv2 environment.

Mine AFNI deconvolved files for sub-bricks/behaviors of interest
and generate a dataframe of voxel beta-coefficients. Dataframes
may be masked by identifying empty coordinates in a group-level
mask.

Dataframes are written for each subject in --subj-list/all, and
a group dataframe can be generated from all subject dataframes.

Subject-level dataframes are titled
    <subj>_<sess>_<task>_desc-<model_name>_betas.tsv
and written to:
    <proj_dir>/data_scanner_BIDS/derivatives/model_afni/<subj>/<sess>/func

The group-level dataframe is written to:
    <proj_dir>/analyses/model_afni/afni_<model_name>_betas.tsv

Examples
--------
afni_extract --sub-list sub-ER0009 sub-ER0016
afni_extract --sub-all

"""
# %%
import os
import sys
import glob
import textwrap
from argparse import ArgumentParser, RawTextHelpFormatter
from func_model.workflows import wf_afni


# %%
def _get_args():
    """Get and parse arguments."""
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="univ",
        help=textwrap.dedent(
            """\
            [univ]
            AFNI model name/type, for triggering different workflows
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
    model_name = args.model_name

    # Check model_name
    if model_name not in ["univ"]:
        print(f"Unsupported model name : {model_name}")
        sys.exit(1)

    # Make subject list, trigger workflow
    if subj_all:
        proj_deriv = os.path.join(
            proj_dir, "data_scanner_BIDS/derivatives/model_afni"
        )
        subj_all = sorted(glob.glob(f"{proj_deriv}/sub-*"))
        subj_list = [os.path.basename(x) for x in subj_all]
    wf_afni.afni_extract(proj_dir, subj_list, model_name)


if __name__ == "__main__":

    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()
