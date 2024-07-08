#!/bin/bash

#SBATCH --job-name=array
#SBATCH --time=200:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

function Usage {
    cat <<USAGE
    Usage: sbatch [OPTS] --array=0-10%2 cli_array.sh -e <sess> -m <model> [-f <path>] [-s "<sub-A> <sub-B>"]

    Schedule SLURM array to execute AFNI deconvolution. Requires execution using
    'sbatch --array=x-y%z' options.

    Optional Arguments:
        -f <fmriprep path>
            Location of directory containing subject fMRIPrep output
        -s "<subj-1> <subj-2>"
            List of subjects with fMRIPrep data on Keoki in quotes

    Required Arguments:
        -e [ses-day2|ses-day3]
            BIDS session identifier
        -m [mixed|task|block]
            Model name for triggering AFNI workflow

    Example Usage:
        sbatch \\
            --output=/work/$(whoami)/EmoRep/logs/afni_array/slurm_%A_%a.log \\
            --array=0-120%10 \\
            array_cli.sh \\
            -e ses-day2 \\
            -m mixed

        sbatch \\
            --output=/work/$(whoami)/EmoRep/logs/afni_array/slurm_%A_%a.log \\
            --array=0-120%10 \\
            array_cli.sh \\
            -e ses-day2 \\
            -m mixed \\
            -s "sub-ER0009 sub-ER0016"

USAGE
}

# Optional arguments
subj_list=()
fmriprep_dir=/mnt/keoki/experiments2/EmoRep/Exp2_Compute_Emotion/data_scanner_BIDS/derivatives/pre_processing/fmriprep

# Capture arguments
while getopts ":e:f:s:m:h" OPT; do
    case $OPT in
    e)
        sess=${OPTARG}
        ;;
    f)
        fmriprep_dir=${OPTARG}
        ;;
    m)
        model=${OPTARG}
        ;;
    s)
        subj_list+=($OPTARG)
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
if [ $OPTIND -lt 5 ]; then
    Usage
    exit 0
fi

# Require arguments
if [ -z $sess ] || [ -z $model ]; then
    Usage
    exit 1
fi

# Check for array
if [ -z $SLURM_ARRAY_TASK_ID ]; then
    Usage
    exit 1
fi

# Stagger job start time
sleep $((RANDOM % 30 + 1))

# Get subject list if needed
if [ -z $subj_list ]; then
    subj_list=($(
        ssh -i $RSA_LS2 \
            $(whoami)@ccn-labarserv2.vm.duke.edu \
            " command ; bash -c 'ls ${fmriprep_dir} | grep sub* | grep -v html'"
    ))
fi

python array_wf_afni.py -e $sess -m $model -s ${subj_list[$SLURM_ARRAY_TASK_ID]}
