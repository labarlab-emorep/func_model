#!/bin/bash

function Usage {
    cat <<USAGE
    Usage: $0 -s <stat-name> [-g group-path] [-t template]

    Generate montages for emotion main effects of afni_lmer output.

    Requires:
        - Template file to exist in group directory
        - Output of func_model's afni_lmer

    Optional Arguments:
        -g <path> = Location of group directory
            (default: /mnt/keoki/experiments2/EmoRep/Exp2_Compute_Emotion/analyses/model_afni_group)
        -t <str> = Template name
            (default: tpl-MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz)

    Required Arguments:
        -s <stat> = Name of stat directory

    Example Usage:
        $0 -s stat-lmer_model-mixed

USAGE
}

# Set default variables
group_dir=/mnt/keoki/experiments2/EmoRep/Exp2_Compute_Emotion/analyses/model_afni_group
tpl_file=tpl-MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz

# Capture arguments
while getopts ":g:s:t:h" OPT; do
    case $OPT in
    g)
        group_dir=${OPTARG}
        ;;
    s)
        stat_name=${OPTARG}
        ;;
    t)
        tpl_file=${OPTARG}
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
if [ $OPTIND -lt 3 ]; then
    Usage
    exit 0
fi

# Require arguments
if [ -z $stat_name ]; then
    echo -e "\nMissing -s option\n" >&2
    Usage
    exit 1
fi

# Validate environment
3dcopy 1>/dev/null 2>&1
if [ $? != 1 ]; then
    echo "AFNI not exectuable in environment" >&2
    exit 1
fi

# Orient to template and stat file, validate
tpl_path=${group_dir}/tpl-MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz
stat_path=${group_dir}/${stat_name}/${stat_name}+tlrc

if [ ! -f $tpl_path ]; then
    echo "Failed to find $tpl_path" >&2
    Usage
    exit 1
fi

if [ ! -f ${stat_path}.HEAD ]; then
    echo "Failed to find ${stat_path}.HEAD " >&2
    Usage
    exit 1
fi

# Set subbrick emotion name, coefficient, and stat IDs
sub_name=(Amu Ang Anx Awe Cal Cra Dis Exc Fea Hor Joy Neu Rom Sad Sur)
sub_coef=(3 9 15 21 27 33 39 45 51 57 63 69 75 81 87)
sub_stat=(4 10 16 22 28 34 40 46 52 58 64 70 76 82 88)

# Validate balanced arrays
if [ ${#sub_name[@]} != ${#sub_id[@]} ] || [ ${#sub_name[@]} != ${#sub_th[@]} ]; then
    echo "Unequal array lengths" >&2
    exit 1
fi

# Setup output directory
out_dir=$(dirname $stat_file)/subbrick_montages
mkdir -p $out_dir

# Make montage for each emotion
c=0
while [ $c -lt ${#sub_name[@]} ]; do
    @chauffeur_afni \
        -ulay $tpl_path \
        -olay $stat_file \
        -set_dicom_xyz 0 10 0 \
        -cbar Reds_and_Blues_Inv \
        -func_range 1 \
        -thr_olay_p2stat 0.001 \
        -thr_olay_pside bisided \
        -set_subbricks -1 ${sub_coef[$c]} ${sub_stat[$c]} \
        -opacity 6 \
        -prefix ${out_dir}/${sub_name[$c]}_mont \
        -montx 5 \
        -monty 5 \
        -set_xhairs OFF \
        -label_mode 1 \
        -label_size 1 \
        -left_is_left YES \
        -label_mode 6 \
        -label_size 2 \
        -box_focus_slices AMASK_FOCUS_ULAY \
        -do_clean

    let c+=1
done
