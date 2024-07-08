#!/bin/bash

#SBATCH --job-name=array
#SBATCH --time=200:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

function Usage {
    cat <<USAGE
    Usage: sbatch cli_array.sh -e <sess> -m <model>

    Schedule SLURM array to execute AFNI deconvolution

    Required Arguments:
        -e [ses-day2|ses-day3]
            BIDS session identifier
        -m [mixed|task|block]
            Model name for triggering AFNI workflow

    Example Usage:
        sbatch \\
            --output=/work/$(whoami)/EmoRep/logs/afni_mixed_array/slurm_%A_%a.log \\
            --array=0-120%10 \\
            array_cli.sh \\
            -e ses-day2 \\
            -m mixed

USAGE
}

# Capture arguments
while getopts ":e:m:h" OPT; do
    case $OPT in
    e)
        sess=${OPTARG}
        ;;
    m)
        model=${OPTARG}
        ;;
    h)
        Usage
        exit 0
        ;;
    :)
        echo -e "\nERROR: option '$OPTARG' missing argument.\n" >&2
        Usage
        exit 1
        ;;
    \?)
        echo -e "\nERROR: invalid option '$OPTARG'.\n" >&2
        Usage
        exit 1
        ;;
    esac
done

# Print help if no args
if [ $OPTIND != 5 ]; then
    Usage
    exit 0
fi

# Require arguments
if [ -z $sess ] || [ -z $model ]; then
    Usage
    exit 1
fi

python array_wf_afni.py -e $sess -m $model
