"""Extract voxel beta weights from FSL FEAT files.

Written for the local labarserv2 environment.

Mine FSL GLM files for contrasts of interest and generate a
dataframe of voxel beta-coefficients. Dataframes may be masked by
identifying coordinates in a group-level mask.

Dataframes are written for each subject in --subj-list/all, and
a group dataframe can be generated from all subject dataframes.

Model names (see fsl_model):
    - lss = conduct full GLMs for every single trial,
            requires --contrast-name tog
    - sep = model stimulus and replay separately,
            requires --contrast-name stim|replay
    - tog = model stimulus and replay together,
            requires --contrast-name tog

Contrast names:
    - replay = replay vs washout
    - stim = stimulus vs washout
    - tog = stim and replay (together) vs washout

Notes
-----
- Subject-level dataframes are titled
    <subj>_<sess>_<task>_<model-level>_<model-name>_betas.tsv
  and written to:
    <proj_dir>/data_scanner_BIDS/derivatives/model_fsl/<subj>/<sess>/func
- The group-level dataframe is titled:
    <model-level>_<model-name>_<contrast>_voxel-betas.tsv
  and written to:
    <proj_dir>/analyses/model_fsl_group

Examples
--------
fsl_extract --sub-all
fsl_extract --sub-list sub-ER0009 sub-ER0016 --contrast-name replay
fsl_extract --sub-all --model-name lss --contrast-name tog

"""
# %%
import os
import sys
import glob
import textwrap
from copy import deepcopy
from argparse import ArgumentParser, RawTextHelpFormatter
from func_model.workflows import wf_fsl


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
        choices=["replay", "stim", "tog"],
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
        choices=["lss", "sep", "tog"],
        help=textwrap.dedent(
            """\
            FSL model name, for triggering different workflows.
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


def _err_msg(model_name: str, con_name: str):
    """Print to user and kill if bad args specified."""
    print(f"Unsupported contrast for model {model_name} : {con_name}")
    sys.exit(1)


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
    if model_name == "tog" and con_name != "tog":
        _err_msg(model_name, con_name)
    if model_name == "sep" and (con_name != "stim" and con_name != "replay"):
        _err_msg(model_name, con_name)
    if model_name == "lss" and con_name != "tog":
        _err_msg(model_name, con_name)

    # Check, make subject list
    proj_deriv = os.path.join(
        proj_dir, "data_scanner_BIDS/derivatives/model_fsl"
    )
    subj_avail = [
        os.path.basename(x) for x in sorted(glob.glob(f"{proj_deriv}/sub-*"))
    ]
    if not subj_avail:
        raise ValueError(f"No FSL output found at : {proj_deriv}")
    if subj_list:
        for subj in subj_list:
            if subj not in subj_avail:
                raise ValueError(
                    "Specified subject not found in "
                    + f"derivatives/model_fsl : {subj}"
                )
    if subj_all:
        subj_list = deepcopy(subj_avail)

    # Submit workflow
    wf_fsl.fsl_extract(proj_dir, subj_list, model_name, model_level, con_name)


if __name__ == "__main__":
    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()
