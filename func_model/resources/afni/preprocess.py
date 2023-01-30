"""Methods for additional preprocessing."""
import os
from func_model.resources.afni import helper
from func_model.resources.general import submit


def smooth_epi(
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


def scale_epi(subj_work, proj_deriv, mask_min, func_preproc, sing_afni):
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
