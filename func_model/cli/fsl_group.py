"""CLI for generating group-level files.

Written for the local labarserv2 environment.

Generate files to help building third- and fourth-level design.fsf
in the FSL GUI. Finds input data for the following GUI paths:
    - FSL > FEAT-FMRI > Data > Select FEAT directories
    - FSL > FEAT-FMRI > Stats > Full model setup > EVs > EV1
    - FSL > FEAT-FMRI > Stats > Full model setup > Contrast & F-tests > EV1

Data for these GUI input locations are written (respectively) to:
    - level-*_name-*_comb-*_task-*_data-input.txt
    - level-*_name-*_comb-*_task-*_stats-evs.txt
    - level-*_name-*_comb-*_task-*_stats-contrasts.txt

Combination names:
    - sess = TODO
    - subj = generate third-level input for combining across subjects
                while maintaining task separate. Generates
                level-third_name-*_comb-subj_task-*.
                Requires -t argument.

Level names:
    - third = TODO
    - fourth = TODO

Examples
--------
fsl_group -l third -c subj --task-name movies
fsl_group -l third -c sess
fsl_group -l fourth -c sess

"""

# %%
import sys
import platform
import textwrap
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
        choices=["sep"],
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
            Path to BIDS-formatted project directory
            (default : %(default)s)
            """
        ),
    )
    parser.add_argument(
        "--task-name",
        help="Value of BIDS task field",
        type=str,
        choices=["scenarios", "movies"],
    )

    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "-c",
        "--comb-name",
        help="Combination type, collapse across a factor/variable",
        type=str,
        choices=["subj", "sess"],
        required=True,
    )
    required_args.add_argument(
        "-l",
        "--model-level",
        help="FSL model level",
        type=str,
        choices=["third", "fourth"],
        required=True,
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
    comb_name = args.comb_name
    model_name = args.model_name
    model_level = args.model_level
    task_name = args.task_name

    # Validate
    if model_level == "third" and comb_name == "subj" and not task_name:
        print(
            "Command 'fsl_group -c subj -l third' missing "
            + "required option : --task-name"
        )
        sys.exit(1)
    if model_level == "fourth" and comb_name != "sess":
        print(
            "Commnad 'fsl_group -l fourth' missing "
            + "required argument : -c sess"
        )
        sys.exit(1)

    # Trigger requested workflow
    if model_level == "third" and comb_name == "subj":
        fsl_subj = wf_fsl.FslThirdSubj(
            proj_dir, model_name=model_name, task=task_name
        )
        fsl_subj.build_input_data()

    if model_level == "third" and comb_name == "sess":
        fsl_sess = wf_fsl.FslThirdSess(proj_dir, model_name=model_name)
        fsl_sess.build_input_data()

    if model_level == "fourth" and comb_name == "sess":
        fsl_sess = wf_fsl.FslFourthSess(proj_dir, model_name=model_name)
        fsl_sess.build_input_data()


if __name__ == "__main__":
    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()
