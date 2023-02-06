"""Methods for additional preprocessing."""
import os
import glob
import fnmatch
from func_model.resources.afni import helper, deconvolve, masks
from func_model.resources.general import submit


def _smooth_epi(
    subj_work,
    proj_deriv,
    func_preproc,
    sing_afni,
    blur_size=3,
):
    """Spatially smooth EPI files.

    Parameters
    ----------
    subj_work : path
        Location of working directory for intermediates
    proj_deriv : path
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories
    func_preproc : list
        Locations of preprocessed EPI files
    sing_afni : path
        Location of AFNI singularity file
    blur_size : int, optional
        Size (mm) of smoothing kernel

    Returns
    -------
    list
        Paths to smoothed EPI files

    Raises
    ------
    TypeError
        Improper parameter types

    """
    # Check arguments
    if not isinstance(blur_size, int):
        raise TypeError("Optional blur_size requires int")
    if not isinstance(func_preproc, list):
        raise TypeError("Argument func_preproc requires list")

    # Start return list, smooth each epi file
    print("\nSmoothing EPI files ...")
    func_smooth = []
    for epi_path in func_preproc:

        # Setup output names/paths, avoid repeating work
        epi_preproc = os.path.basename(epi_path)
        desc_preproc = epi_preproc.split("desc-")[1].split("_")[0]
        epi_smooth = epi_preproc.replace(desc_preproc, "smoothed")
        out_path = os.path.join(subj_work, epi_smooth)
        if os.path.exists(out_path):
            func_smooth.append(out_path)
            continue

        # Smooth data
        print(f"\tStarting smoothing of {epi_path}")
        bash_list = [
            "3dmerge",
            f"-1blur_fwhm {blur_size}",
            "-doall",
            f"-prefix {out_path}",
            epi_path,
        ]
        sing_prep = helper.prepend_afni_sing(proj_deriv, subj_work, sing_afni)
        bash_cmd = " ".join(sing_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_path, "Smooth run")

        # Update return list
        func_smooth.append(out_path)

    # Double-check correct order of files
    func_smooth.sort()
    return func_smooth


def _scale_epi(subj_work, proj_deriv, mask_min, func_preproc, sing_afni):
    """Scale EPI timeseries.

    Parameters
    ----------
    subj_work : path
        Location of working directory for intermediates
    proj_deriv : path
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories
    mask_min : path
        Location of minimum-value mask, output of
        afni.MakeMasks.minimum
    func_preproc : list
        Locations of preprocessed EPI files
    sing_afni : path
        Location of AFNI singularity file

    Returns
    -------
    list
        Paths to scaled EPI files

    Raises
    ------
    TypeError
        Improper parameter types

    """
    # Check arguments
    if not isinstance(func_preproc, list):
        raise TypeError("Argument func_preproc requires list")

    # Start return list, scale each epi file supplied
    print("\nScaling EPI files ...")
    func_scaled = []
    for epi_path in func_preproc:

        # Setup output names, avoid repeating work
        epi_preproc = os.path.basename(epi_path)
        desc_preproc = epi_preproc.split("desc-")[1].split("_")[0]
        epi_tstat = "tmp_" + epi_preproc.replace(desc_preproc, "tstat")
        out_tstat = os.path.join(subj_work, epi_tstat)
        epi_scale = epi_preproc.replace(desc_preproc, "scaled")
        out_path = os.path.join(subj_work, epi_scale)
        if os.path.exists(out_path):
            func_scaled.append(out_path)
            continue

        # Determine mean values
        print(f"\tStarting scaling of {epi_path}")
        bash_list = [
            "3dTstat",
            f"-prefix {out_tstat}",
            epi_path,
        ]
        sing_prep = helper.prepend_afni_sing(proj_deriv, subj_work, sing_afni)
        bash_cmd = " ".join(sing_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_tstat, "Tstat run")

        # Scale values
        bash_list = [
            "3dcalc",
            f"-a {epi_path}",
            f"-b {out_tstat}",
            f"-c {mask_min}",
            "-expr 'c * min(200, a/b*100)*step(a)*step(b)'",
            f"-prefix {out_path}",
        ]
        bash_cmd = " ".join(sing_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_path, "Scale run")

        # Update return list
        func_scaled.append(out_path)

    # Double-check correct order of files
    func_scaled.sort()
    return func_scaled


def extra_preproc(subj, sess, subj_work, proj_deriv, sing_afni, do_rest=False):
    """Conduct extra preprocessing for AFNI.

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
    do_rest : bool
        Whether to work with resting state or task EPI data

    Returns
    -------
    tuple
        [0] = dictionary of functional files
        [1] = dictionary of anatomical files

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

    # Find task or rest motion files from fMRIPrep
    subj_func_fp = os.path.join(subj_deriv_fp, sess, "func")
    if do_rest:
        mot_files = glob.glob(f"{subj_func_fp}/*task-rest*timeseries.tsv")
    else:
        mot_files = [
            x
            for x in glob.glob(f"{subj_func_fp}/*timeseries.tsv")
            if not fnmatch.fnmatch(x, "*task-rest*")
        ]
    mot_files.sort()
    if not mot_files:
        raise FileNotFoundError(
            "Expected to find fmriprep files func/*timeseries.tsv"
        )

    # Find preprocessed EPI files, task or resting
    if do_rest:
        run_files = glob.glob(
            f"{subj_func_fp}/*task-rest*_res-2_desc-preproc_bold.nii.gz"
        )
    else:
        run_files = [
            x
            for x in glob.glob(
                f"{subj_func_fp}/*_res-2_desc-preproc_bold.nii.gz"
            )
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
    make_masks = masks.MakeMasks(
        subj_work, proj_deriv, anat_dict, func_dict, sing_afni
    )
    anat_dict["mask-int"] = make_masks.intersect()
    tiss_masks = make_masks.tissue()
    anat_dict["mask-WMe"] = tiss_masks["WM"]
    anat_dict["mask-CSe"] = tiss_masks["CS"]
    anat_dict["mask-min"] = make_masks.minimum()

    # Smooth and scale EPI data
    smooth_epi = _smooth_epi(
        subj_work, proj_deriv, func_dict["func-preproc"], sing_afni
    )
    func_dict["func-scaled"] = _scale_epi(
        subj_work,
        proj_deriv,
        anat_dict["mask-min"],
        smooth_epi,
        sing_afni,
    )

    # Make AFNI-style motion and censor files
    make_motion = deconvolve.MotionCensor(
        subj_work, proj_deriv, func_dict["func-motion"], sing_afni
    )
    func_dict["mot-mean"] = make_motion.mean_motion()
    func_dict["mot-deriv"] = make_motion.deriv_motion()
    func_dict["mot-cens"] = make_motion.censor_volumes()
    _ = make_motion.count_motion()

    return (func_dict, anat_dict)
