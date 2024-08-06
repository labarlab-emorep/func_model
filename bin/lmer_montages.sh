#!/bin/bash

function Usage {
    cat <<USAGE
    Usage: $0 -s <stat-name> [-c int] [-g group-path] [-n int] [-t template]

    Generate montages and cluster tables for emotion main effects of
    afni_lmer output.

    PNG montage files are written to
        <group-path>/<stat-name>/emotion_montages
    and TXT tables are written to
        <group-path>/<stat-name>/emotion_tables

    Requires:
        - Template file to exist in <group-path>
        - Output of func_model's afni_lmer

    Optional Arguments:
        -c <int> = Clusterize threshold from 3dClustSim for bisided p=0.001
            (default: 12)
        -g <path> = Location of group directory
            (default: /mnt/keoki/experiments2/EmoRep/Exp2_Compute_Emotion/analyses/model_afni_group)
        -n [1|2|3] = Nearest neighbors
            (default: 2)
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
n_nbr=2
c_thr=12

# Capture arguments
while getopts ":c:g:n:s:t:h" OPT; do
    case $OPT in
    c)
        c_thr=${OPTARG}
        ;;
    g)
        group_dir=${OPTARG}
        ;;
    n)
        n_nbr=${OPTARG}
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
if [ $? != 0 ]; then
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
sub_name=(
    Amusement Anger Anxiety Awe Calmness
    Craving Disgust Excitement Fear Horror
    Joy Neutral Romance Sadness Surprise
)
sub_coef=(3 9 15 21 27 33 39 45 51 57 63 69 75 81 87)
sub_stat=(4 10 16 22 28 34 40 46 52 58 64 70 76 82 88)

# Validate balanced arrays
if [ ${#sub_name[@]} != ${#sub_coef[@]} ] || [ ${#sub_name[@]} != ${#sub_stat[@]} ]; then
    echo "Unequal array lengths" >&2
    exit 1
fi

# Setup output directories
mnt_dir=$(dirname $stat_path)/emotion_montages
tbl_dir=$(dirname $stat_path)/emotion_tables
mkdir -p $mnt_dir $tbl_dir

c=0
while [ $c -lt ${#sub_name[@]} ]; do

    # Make montage for each emotion
    echo -e "\n\nDrawing ${sub_name[$c]} montages to:\n\t${mnt_dir}\n\n" >&1
    @chauffeur_afni \
        -ulay $tpl_path \
        -olay $stat_path \
        -set_dicom_xyz 0 10 0 \
        -cbar Reds_and_Blues_Inv \
        -func_range 1 \
        -thr_olay_p2stat 0.001 \
        -thr_olay_pside bisided \
        -set_subbricks -1 ${sub_coef[$c]} ${sub_stat[$c]} \
        -opacity 6 \
        -prefix ${mnt_dir}/${sub_name[$c]} \
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

    # Make a cluster table for each emotion
    echo -e "\n\nWriting ${sub_name[$c]} tables to:\n\t${tbl_dir}\n\n" >&1
    3dClusterize \
        -nosum \
        -1Dformat \
        -inset $stat_path \
        -idat ${sub_coef[$c]} \
        -ithr ${sub_stat[$c]} \
        -orient LPI \
        -NN $n_nbr \
        -clust_nvox $c_thr \
        -bisided p=0.001 \
        >${tbl_dir}/${sub_name[$c]}_table.txt

    let c+=1
done
