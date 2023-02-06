"""Title.

Desc.

Model names:
    - movies = model group response to emotion movies versus baseline,
                used for sanity checking
    - scenarios = TODO model group response to emotion scenarios versus
                baseline, used for sanity checking
    - stim = TODO model group response to emotional stimuli versus washout

Examples
--------
univ_afni -n movies

"""
# %%
import os
import sys
import textwrap
from datetime import datetime
from argparse import ArgumentParser, RawTextHelpFormatter
from func_model.resources.general import submit
from func_model.resources.afni import helper


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

    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "-n",
        "--model-name",
        help=textwrap.dedent(
            """\
            [movies | scenarios]
            Stimulus type to model at group level.
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

    # Check model_name
    if model_name not in ["movies"]:
        print(f"Unsupported model name : {model_name}")
        sys.exit(1)

    # Setup group project directory, paths
    proj_deriv = os.path.join(proj_dir, "derivatives")
    proj_rawdata = os.path.join(proj_dir, "rawdata")

    # Get environmental vars
    sing_afni = os.environ["SING_AFNI"]
    user_name = os.environ["USER"]

    # Setup work directory, for intermediates
    work_deriv = os.path.join("/work", user_name, "EmoRep")
    # now_time = datetime.now()
    # log_dir = os.path.join(
    #     work_deriv,
    #     f"logs/func-afni_model-{model_name}_"
    #     + f"{now_time.strftime('%Y-%m-%d_%H:%M')}",
    # )
    log_dir = os.path.join(
        work_deriv, f"logs/func-afni_model-{model_name}_test"
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # submit per emo
    emo_dict = helper.emo_switch()

    # workflows.afni_extract(proj_dir, subj_list, model_name)


if __name__ == "__main__":

    # Require proj env
    env_found = [x for x in sys.path if "emorep" in x]
    if not env_found:
        print("\nERROR: missing required project environment 'emorep'.")
        print("\tHint: $labar_env emorep\n")
        sys.exit(1)
    main()
