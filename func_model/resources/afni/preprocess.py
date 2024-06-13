"""Methods for additional preprocessing."""

import os
import glob
import fnmatch
from typing import Union
from func_model.resources.afni import helper as afni_helper
from func_model.resources.afni import masks
from func_model.resources.general import submit


def _smooth_epi(
    subj_work: Union[str, os.PathLike],
    work_deriv: Union[str, os.PathLike],
    func_preproc: list,
    blur_size: int = 3,
):
    """Spatially smooth EPI files."""
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
        sing_prep = afni_helper.prepend_afni_sing(work_deriv, subj_work)
        bash_cmd = " ".join(sing_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_path, "Smooth run")

        # Update return list
        func_smooth.append(out_path)

    # Double-check correct order of files
    func_smooth.sort()
    return func_smooth


def _scale_epi(
    subj_work: Union[str, os.PathLike],
    work_deriv: Union[str, os.PathLike],
    mask_min: Union[str, os.PathLike],
    func_preproc: list,
) -> list:
    """Scale EPI timeseries."""
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
        sing_prep = afni_helper.prepend_afni_sing(work_deriv, subj_work)
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


def _get_fmriprep(
    subj: str, sess: str, work_deriv: Union[str, os.PathLike], do_rest: bool
):
    """Find required fMRIPrep output."""
    # Set search dictionary used to make sess_anat
    #   key = identifier of file in sess_anat
    #   value = searchable string by glob
    get_sess_anat = {
        "anat-preproc": "desc-preproc_T1w",
        "mask-brain": "desc-brain_mask",
        "mask-probCS": "label-CSF_probseg",
        "mask-probGM": "label-GM_probseg",
        "mask-probWM": "label-WM_probseg",
    }

    # Start dictionary of anatomical files
    sess_anat = {}
    subj_deriv_fp = os.path.join(work_deriv, "fmriprep", subj, sess)
    for key, search in get_sess_anat.items():
        file_path = sorted(
            glob.glob(
                f"{subj_deriv_fp}/anat/{subj}_*space-*_{search}.nii.gz",
            )
        )
        if not file_path:
            raise FileNotFoundError("Missing expected fMRIPrep anat output")
        sess_anat[key] = file_path[0]

    # Find task or rest motion files from fMRIPrep
    subj_func_fp = os.path.join(subj_deriv_fp, "func")
    if do_rest:
        mot_files = glob.glob(f"{subj_func_fp}/*task-rest*timeseries.tsv")
    else:
        mot_files = [
            x
            for x in glob.glob(f"{subj_func_fp}/*timeseries.tsv")
            if not fnmatch.fnmatch(x, "*task-rest*")
        ]
    if not mot_files:
        raise FileNotFoundError(
            "Expected to find fmriprep files func/*timeseries.tsv"
        )
    mot_files.sort()

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
    if not run_files:
        raise FileNotFoundError(
            "Expected to find fmriprep files *res-2_desc-preproc_bold.nii.gz"
        )
    run_files.sort()

    # Check that each preprocessed file has a motion file
    if len(run_files) != len(mot_files):
        raise ValueError(
            f"""
        Length of motion files differs from run files.
            motion files : {mot_files}
            run files : {run_files}
        """
        )

    return (mot_files, run_files, sess_anat)


def extra_preproc(subj, sess, subj_work, work_deriv, do_rest=False):
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
    subj_work : str, os.PathLike
        Location of working directory for intermediates
    work_deriv : str, os.PathLike
        Location of intermediate derivatives
    do_rest : bool
        Whether to work with resting state or task EPI data

    Returns
    -------
    tuple
        [0] = dictionary of functional files
        [1] = dictionary of anatomical files

    """
    # Find required files
    mot_files, run_files, sess_anat = _get_fmriprep(
        subj, sess, work_deriv, do_rest
    )

    # Start dictionary of EPI files
    sess_func = {}
    sess_func["func-motion"] = mot_files
    sess_func["func-preproc"] = run_files

    # Make required masks
    make_masks = masks.MakeMasks(subj_work, work_deriv, sess_anat, sess_func)
    sess_anat["mask-int"] = make_masks.intersect()
    tiss_masks = make_masks.tissue()
    sess_anat["mask-WMe"] = tiss_masks["WM"]
    sess_anat["mask-CSe"] = tiss_masks["CS"]
    sess_anat["mask-min"] = make_masks.minimum()

    # Smooth and scale EPI data
    smooth_epi = _smooth_epi(subj_work, work_deriv, sess_func["func-preproc"])
    sess_func["func-scaled"] = _scale_epi(
        subj_work,
        work_deriv,
        sess_anat["mask-min"],
        smooth_epi,
    )
    return (sess_func, sess_anat)
