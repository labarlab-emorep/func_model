"""Conduct multivariate testing using AFNI-based methods.

Written for the local labarserv2 environment.

Construct and execute simple multivariate tests for sanity checking
pipeline output. Output is written to:
    <proj-dir>/analyses/model_afni/mvm_<model-name>

Model names:
    - rm = repeated-measures ANOVA using two within-subject factors.
            Factor A is session stimulus (movies, scenarios), and
            Factor B is stimulus type (emotion, washout). Main and
            interactive effects are generated as well as an
            emotion-washout T-stat.

Examples
--------
afni_mvm -n rm
afni_mvm -n rm --emo-name fear disgust

"""
# %%
import sys
import textwrap
from argparse import ArgumentParser, RawTextHelpFormatter
from func_model.workflows import wf_afni
from func_model.resources.afni import helper as afni_helper


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
        "--emo-name",
        nargs="+",
        help=textwrap.dedent(
            """\
            List of emotions to test specifically
            (default : None -- all emotions tested)
            """
        ),
        type=str,
    )

    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "-n",
        "--model-name",
        help=textwrap.dedent(
            """\
            [rm]
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
    """Capture, validate arguments and submit workflow."""
    args = _get_args().parse_args()
    proj_dir = args.proj_dir
    model_name = args.model_name
    emo_list = args.emo_name
    if not afni_helper.valid_mvm_test(model_name):
        print(f"Unsupported model name : {model_name}")
        sys.exit(1)

    # Get, validate emotion list
    emo_dict = afni_helper.emo_switch()
    if emo_list:
        for emo in emo_list:
            if emo not in emo_dict.keys():
                raise ValueError(f"Invalid emotion specified : {emo}")

    # Submit workflow for each emotion
    emo_iter = emo_list if emo_list else emo_dict.keys()
    for emo_name in emo_iter:
        wf_afni.afni_mvm(proj_dir, model_name, emo_name)


if __name__ == "__main__":
    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()

# %%
