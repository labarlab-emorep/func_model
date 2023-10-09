"""CLI for generating group-level files.

Written for the local labarserv2 environment.

Generate files to help building third- and fourth-level design.fsf
in the FSL GUI. Finds input data for the following GUI paths:
    - FSL > FEAT-FMRI > Data > Select FEAT directories
    - FSL > FEAT-FMRI > Stats > Full model setup > EVs > EV1
    - FSL > FEAT-FMRI > Stats > Full model setup > Contrast & F-tests > EV1

Data for these GUI input locations are written (respectively) to:
    - level-third_name-*_comb-*_task-*_data-input.txt
    - level-third_name-*_comb-*_task-*_stats-evs.txt
    - level-third_name-*_comb-*_task-*_stats-contrasts.txt

Combination names:
    - subj = generate third-level input for combining across subjects
                while maintaining task separate. Generates
                level-third_name-*_comb-subj_task-*.
                Requires -t argument.

Examples
--------
fsl_group -c subj -t movies
fsl_group -c subj -t scenarios

"""

# %%
import sys
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

    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "-c",
        "--comb-name",
        help="Combination type",
        type=str,
        choices=["subj"],
        required=True,
    )
    required_args.add_argument(
        "-t",
        "--task-name",
        help="Value of BIDS task field",
        type=str,
        choices=["scenarios", "movies"],
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
    comb_name = args.comb_name
    model_name = args.model_name
    task_name = args.task_name

    if comb_name == "subj":
        fsl_subj = wf_fsl.FslThirdSubj(
            proj_dir, model_name=model_name, task=task_name
        )
        fsl_subj.build_input_data()
        fsl_subj.write_out()


if __name__ == "__main__":

    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()
