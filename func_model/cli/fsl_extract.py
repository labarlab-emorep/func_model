"""Extract voxel beta weights from FSL FEAT files.

Written for the local labarserv2 environment.

Mine FSL GLM files for contrasts of interest and generate a
dataframe of voxel beta-coefficients. Dataframes may be masked by
identifying coordinates in a group-level mask. Extracted beta-values
are written sent to MySQL:
    db_emorep.tbl_betas_*

Model names (see fsl_model):
    - lss = conduct full GLMs for every single trial
    - sep = model stimulus and replay separately

Notes
-----
- Extraction of betas for model name 'tog' has been deprecated.

Examples
--------
fsl_extract --sub-all
fsl_extract --sub-list sub-ER0009 sub-ER0016 --overwrite
fsl_extract --sub-all --model-name lss

"""
# %%
import os
import sys
import glob
import platform
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
        "--model-name",
        type=str,
        default="sep",
        choices=["lss", "sep"],
        help=textwrap.dedent(
            """\
            FSL model name, for triggering different workflows.
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing records",
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
    """Trigger workflow."""
    # Check env
    if "labarserv2" not in platform.uname().node:
        print("fsl_extract is required to run on labarserv2.")
        sys.exit(1)

    # Get cli args
    args = _get_args().parse_args()
    subj_list = args.sub_list
    subj_all = args.sub_all
    proj_dir = args.proj_dir
    model_name = args.model_name
    overwrite = args.overwrite

    # Assign contrast name (now that tog, replay are deprecated)
    if model_name == "lss":
        con_name = "tog"
    elif model_name == "sep":
        con_name = "stim"

    # Check, make subject list
    proj_deriv = os.path.join(
        proj_dir, "data_scanner_BIDS/derivatives/model_fsl"
    )
    subj_avail = [
        os.path.basename(x) for x in sorted(glob.glob(f"{proj_deriv}/sub-*"))
    ]
    if not subj_avail:
        raise ValueError(f"No FSL output found at : {proj_deriv}")
    if subj_all:
        subj_list = deepcopy(subj_avail)

    # Submit workflow
    ex_reg = wf_fsl.ExtractBetas(
        proj_dir, subj_list, model_name, con_name, overwrite
    )
    ex_reg.get_betas()


if __name__ == "__main__":
    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()

# %%
