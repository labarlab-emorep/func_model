#!/bin/bash

function Usage {
    cat <<USAGE
    Usage: ./array_cli.sh -e <sess> -m <model> [-a <path> -f <path>]

    Schedule array_submit.sh with SLURM scheduler as array of jobs. Finds
    participants with fMRIPrep output and missing model_afni decon output.

    Optional Arguments:
        -a <path>
            Keoki location of afni_model output
        -f <path>
            Keoki location of fMRIPrep output

    Required Arguments:
        -e [ses-day2|ses-day3]
            BIDS session identifier
        -m [mixed|task|block]
            Model name for triggering AFNI workflow

    Example Usage:
        ./array_cli.sh \\
            -e ses-day2 \\
            -m mixed

USAGE
}

# Optional args
deriv_dir=/mnt/keoki/experiments2/EmoRep/Exp2_Compute_Emotion/data_scanner_BIDS/derivatives
fmriprep_dir=${deriv_dir}/pre_processing/fmriprep
afni_dir=${deriv_dir}/model_afni

# Capture arguments
while getopts ":a:e:f:m:h" OPT; do
    case $OPT in
    a)
        afni_dir=${OPTARG}
        ;;
    e)
        sess=${OPTARG}
        ;;
    f)
        fmriprep_dir=${OPTARG}
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

# Require arguments
if [ -z $sess ] || [ -z $model ]; then
    Usage
    exit 1
fi

# Setup output location
log_dir=/work/$(whoami)/EmoRep/logs/afni_array
mkdir -p $log_dir

# Find subjects with fMRIPrep output
echo "Building list of subjects ..."
fmriprep_list=(
    $(
        ssh -i $RSA_LS2 \
            $(whoami)@ccn-labarserv2.vm.duke.edu \
            " command ; bash -c 'ls ${fmriprep_dir} 2>/dev/null | grep sub* | grep -v html'"
    )
)

# Find subjects with afni_model decon output
search_str="${afni_dir}/sub-*/$sess/func/sub-*_${sess}_\
task-*_desc-decon_model-${model}_stats_REML+tlrc.HEAD"
decon_list=(
    $(
        ssh -i $RSA_LS2 \
            $(whoami)@ccn-labarserv2.vm.duke.edu \
            " command ; bash -c 'ls ${search_str} 2>/dev/null'"
    )
)

# Determine subject in fmriprep and not in decon lists
for subj in ${fmriprep_list[@]}; do
    printf '%s\n' "${decon_list[@]}" |
        grep $subj 1>/dev/null 2>&1 ||
        subj_list+=($subj)
done
echo -e "\tDone"

# Check that subjs were found
num_subj=${#subj_list[@]}
if [ $num_subj == 0 ]; then
    echo -e "\nNo subjects detected! Looking in:" >&2
    echo -e "\tfmriprep_dir=${fmriprep_dir}" >&2
    echo -e "\tafni_dir=${afni_dir}\n" >&2
    Usage
    exit 1
fi
echo -e "\tFound ${num_subj} subjects"

# Schedule array
sbatch \
    --output=${log_dir}/slurm_%A_%a.log \
    --array=0-${num_subj}%10 \
    array_submit.sh \
    -e $sess \
    -m $model \
    "${subj_list[@]}"
