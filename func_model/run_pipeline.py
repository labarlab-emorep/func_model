"""Run processing methods for workflow."""
# %%
import os
import glob
import fnmatch
from func_model import afni


# %%
def afni_sanity_tfs(subj, sess, subj_work, subj_sess_raw):
    """Make timing files for sanity check.

    Generate a set of AFNI-styled timing files in order to
    check the design and manipulation.

    Timing files for common, session stimulus, and selection
    tasks are generated. Fixations are excluded to serve as
    model baseline.

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    subj_work : path
        Location of working directory for generating intermediate files
    subj_sess_raw : path
        Location of participant's session rawdata, used to find
        BIDS event files

    Returns
    -------
    dict
        key = event name
        value = path to timing file

    Raises
    ------
    FileNotFoundError
        BIDS event files are missing
    ValueError
        Unexpected task name

    """
    # Find events files
    sess_events = sorted(glob.glob(f"{subj_sess_raw}/func/*events.tsv"))
    if not sess_events:
        raise FileNotFoundError(
            f"Expected BIDs events files in {subj_sess_raw}"
        )

    # Identify and validate task name
    task = os.path.basename(sess_events[0]).split("task-")[-1].split("_")[0]
    task_valid = ["movies", "scenarios"]
    if task not in task_valid:
        raise ValueError(f"Expected task names movies|scenarios, found {task}")

    # Generate timing files
    make_tf = afni.TimingFiles(subj, sess, task, subj_work, sess_events)
    tf_com = make_tf.common_events()
    tf_sess = make_tf.session_events()
    tf_sel = make_tf.select_events()
    tf_all = tf_com + tf_sess + tf_sel

    # Setup output dict
    sess_tfs = {}
    for tf_path in tf_all:
        h_key = os.path.basename(tf_path).split("desc-")[1].split("_")[0]
        sess_tfs[h_key] = tf_path

    return sess_tfs


# %%
def afni_sanity_preproc(subj, sess, subj_work, proj_deriv, sing_afni):
    """Conduct extra preprocessing for AFNI sanity check.

    Identify required files from fMRIPrep and FSL, then conduct
    extra preprocessing to ready for AFNI deconvolution. The
    pipeline steps are:
        -   Make intersection mask
        -   Make eroded CSF, WM masks
        -   Make minimum-value mask
        -   Spatially smooth EPI
        -   Scale EPI timeseries
        -   Make mean, derivative motion files
        -   Make censor, inverted censor files

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    subj_work : path
        Location of working directory for intermediates
    proj_deriv : path
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories
    sing_afni : path
        Location of AFNI singularity file

    Returns
    -------
    tuple
        [0] = dictionary of functional files, "key": value description:
            "func-motion": [/paths/to/fMRIPrep/timeseries.tsv]
            "func-preproc": [/paths/to/fsl/denoised/nii]
            "func-scaled": [/paths/to/scaled/nii]
            "func-mean": /path/to/mean/motion/file
            "func-deriv": /path/to/derivative/motion/file
            "func-cens": /path/to/censor/file
            "func-inv": /path/to/inverted/censor/file
        [1] = dictionary of anatomical files, "key": value description:
            "anat-preproc": /path/to/preprocessed/t1w/nii
            "mask-brain": /path/to/anat/brain/mask
            "mask-probCS": /path/to/probabilistic/csf/label
            "mask-probGM": /path/to/probabilistic/gm/label
            "mask-probWM": /path/to/probabilistic/WM/label
            "mask-int": /path/to/epi-anat/intersection/mask
            "mask-WMe": /path/to/eroded/wm/mask
            "mask-CSe": /path/to/eroded/csf/mask
            "mask-min": /path/to/minimum/value/mask

    Raises
    ------
    FileNotFoundError
        Missing anatomical or functional fMRIPrep output
        Missing functional FSL output
    ValueError
        Each preprocessed EPI file is not paired with an
        fMRIPrep timeseries.tsv

    """
    # Set search dictionary used to make anat_dict
    #   key = identifier of file in anat_dict
    #   value = searchable string by glob
    get_anat_dict = {
        "anat-preproc": "desc-preproc_T1w",
        "mask-brain": "desc-brain_mask",
        "mask-probCS": "label-CSF_probseg",
        "mask-probGM": "label-GM_probseg",
        "mask-probWM": "label-WM_probseg",
    }

    # Start dictionary of anatomical files
    anat_dict = {}
    subj_deriv_fp = os.path.join(proj_deriv, f"pre_processing/fmriprep/{subj}")
    for key, search in get_anat_dict.items():
        file_path = sorted(
            glob.glob(
                f"{subj_deriv_fp}/**/anat/{subj}_*space-*_{search}.nii.gz",
                recursive=True,
            )
        )
        if file_path:
            anat_dict[key] = file_path[0]
        else:
            raise FileNotFoundError(
                f"Expected to find fmriprep file anat/*_{search}.nii.gz"
            )

    # Find motion files from fMRIPrep
    mot_files = [
        x
        for x in glob.glob(f"{subj_deriv_fp}/{sess}/func/*timeseries.tsv")
        if not fnmatch.fnmatch(x, "*task-rest*")
    ]
    mot_files.sort()
    if not mot_files:
        raise FileNotFoundError(
            "Expected to find fmriprep files func/*timeseries.tsv"
        )

    # Find preprocessed EPI files
    # subj_deriv_fsl = os.path.join(
    #     proj_deriv, f"pre_processing/fsl_denoise/{subj}/{sess}/func"
    # )
    # run_files = [
    #     x
    #     for x in glob.glob(f"{subj_deriv_fsl}/*tfiltMasked_bold.nii.gz")
    #     if not fnmatch.fnmatch(x, "*task-rest*")
    # ]
    # run_files.sort()
    # if not run_files:
    #     raise FileNotFoundError(
    #         "Expected to find fsl_denoise files *tfiltMasked_bold.nii.gz"
    #     )

    # Find preprocessed EPI files
    subj_func_fp = os.path.join(subj_deriv_fp, sess, "func")
    run_files = [
        x
        for x in glob.glob(f"{subj_func_fp}/*_res-2_desc-preproc_bold.nii.gz")
        if not fnmatch.fnmatch(x, "*task-rest*")
    ]
    run_files.sort()
    if not run_files:
        raise FileNotFoundError(
            "Expected to find fmriprep files *res-2_desc-preproc_bold.nii.gz"
        )

    # Check that each preprocessed file has a motion file
    if len(run_files) != len(mot_files):
        raise ValueError(
            f"""
        Length of motion files differs from run files.
            motion files : {mot_files}
            run files : {run_files}
        """
        )

    # Start dictionary of EPI files
    func_dict = {}
    func_dict["func-motion"] = mot_files
    func_dict["func-preproc"] = run_files

    # Make required masks
    make_masks = afni.MakeMasks(
        subj, sess, subj_work, proj_deriv, anat_dict, func_dict, sing_afni
    )
    anat_dict["mask-int"] = make_masks.intersect()
    tiss_masks = make_masks.tissue()
    anat_dict["mask-WMe"] = tiss_masks["WM"]
    anat_dict["mask-CSe"] = tiss_masks["CS"]
    anat_dict["mask-min"] = make_masks.minimum()

    # Smooth and scale EPI data
    smooth_epi = afni.smooth_epi(
        subj_work, proj_deriv, func_dict["func-preproc"], sing_afni
    )
    func_dict["func-scaled"] = afni.scale_epi(
        subj_work,
        proj_deriv,
        anat_dict["mask-min"],
        smooth_epi,
        sing_afni,
    )

    # Make AFNI-style motion and censor files
    make_motion = afni.MotionCensor(
        subj_work, proj_deriv, func_dict["func-motion"]
    )
    func_dict["func-mean"] = make_motion.mean_motion()
    func_dict["func-deriv"] = make_motion.deriv_motion()
    mot_cens, mot_cens_inv = make_motion.censor_volumes()
    func_dict["func-cens"] = mot_cens
    func_dict["func-inv"] = mot_cens_inv
    _ = make_motion.count_motion()

    return (func_dict, anat_dict)
