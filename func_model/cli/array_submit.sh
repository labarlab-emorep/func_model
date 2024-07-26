#!/bin/bash

#SBATCH --job-name=array
#SBATCH --time=200:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G

function Usage {
    cat <<USAGE
    Usage: sbatch [OPTS] array_submit.sh -e <sess> -m <model> sub-1 [sub-2 sub-N]

    Schedule SLURM array to execute AFNI deconvolution.

    Requires:
        - Last argument(s) as BIDS subject IDs
        - Execution using 'sbatch --array=x-y%z' option.

    Required Arguments:
        -e [ses-day2|ses-day3]
            BIDS session identifier
        -m [mixed|task|block]
            Model name for triggering AFNI workflow

    Example Usage:
        sbatch \\
            --output=/path/to/log/slurm_%A_%a.log \\
            --array=0-120%10 \\
            array_submit.sh \\
            -e ses-day2 \\
            -m mixed \\
            sub-1 sub-2 .. sub-120

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
if [ $OPTIND -lt 5 ]; then
    Usage
    exit 0
fi

# Get subject list from remaining parameters
shift 4
subj_list=("$@")
if [ ${#subj_list[@]} == 0 ]; then
    echo "ERROR: Empty subject list"
    Usage
    exit 1
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
python array_wf_afni.py -e $sess -m $model -s ${subj_list[$SLURM_ARRAY_TASK_ID]}
