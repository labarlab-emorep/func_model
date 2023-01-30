"""Pipelines supporting AFNI and FSL."""
# %%
import os
import glob
from func_model.resources.afni import helper, afni_pipelines
from func_model.resources.afni import deconvolve, masks, group
from func_model.resources.fsl import fsl, fsl_pipelines


# %%
def afni_task(
    subj,
    sess,
    proj_rawdata,
    proj_deriv,
    work_deriv,
    sing_afni,
    model_name,
    log_dir,
):
    """Conduct AFNI-based deconvolution for sanity checking.

    Sanity check - model processing during movie|scenario presenation.
    Supplies high-level steps, coordinates actual work in afni_pipelines
    and afni modules.

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    proj_rawdata : path
        Location of BIDS-organized project rawdata
    proj_deriv : path
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories
    work_deriv : path
        Parent location for writing pipeline intermediates
    sing_afni : path
        Location of AFNI singularity file
    model_name : str
        [univ | mixed]
        Desired AFNI model, for triggering different workflows
    log_dir : path
        Output location for log files and scripts

    Returns
    -------
    triple
        [0] = dictionary of timing files
        [1] = dictionary of anat files
        [2] = dictionary of func files

    Raises
    ------
    ValueError
        Model name/type not supported

    """
    # Validate
    if model_name not in ["univ", "mixed"]:
        raise ValueError(f"Unsupported model name : {model_name}")

    # Check that session exists for participant
    subj_sess_raw = os.path.join(proj_rawdata, subj, sess)
    if not os.path.exists(subj_sess_raw):
        print(f"Directory not detected : {subj_sess_raw}\n\tSkipping.")

    # Setup output directory
    subj_work = os.path.join(work_deriv, "model_afni-task", subj, sess, "func")
    if not os.path.exists(subj_work):
        os.makedirs(subj_work)

    # Extra pre-processing steps
    sess_func, sess_anat = afni_pipelines.extra_preproc(
        subj, sess, subj_work, proj_deriv, sing_afni
    )

    # Generate timing files - find appropriate pipeline for model_name
    pipe_mod = __import__(
        "func_model.resources.afni.afni_pipelines",
        fromlist=[f"make_{model_name}_tfs"],
    )
    tf_pipe = getattr(pipe_mod, f"make_{model_name}_tfs")
    sess_timing = tf_pipe(subj, sess, subj_work, subj_sess_raw)

    # Generate deconvolution command
    write_decon = deconvolve.WriteDecon(
        subj_work,
        proj_deriv,
        sess_func,
        sess_anat,
        sing_afni,
    )
    write_decon.build_decon(model_name, sess_tfs=sess_timing)

    # Use decon command to make REMl command, execute REML
    make_reml = deconvolve.RunReml(
        subj_work,
        proj_deriv,
        sess_anat,
        sess_func,
        sing_afni,
        log_dir,
    )
    reml_path = make_reml.generate_reml(
        subj, sess, write_decon.decon_cmd, write_decon.decon_name
    )
    sess_func["func-decon"] = make_reml.exec_reml(
        subj, sess, reml_path, write_decon.decon_name
    )

    # Clean
    helper.MoveFinal(subj, sess, proj_deriv, subj_work, sess_anat, model_name)
    return (sess_timing, sess_anat, sess_func)


def afni_rest(
    subj,
    sess,
    proj_rawdata,
    proj_deriv,
    work_deriv,
    sing_afni,
    model_name,
    log_dir,
):
    """Conduct AFNI-styled resting state analysis for sanity checking.

    Based on example 11 of afni_proc.py and s17.proc.FT.rest.11
    of afni_data6. Use 3ddeconvolve to generate a no-censor matrix,
    then project correlation matrix accounting for WM and CSF
    timeseries. Then generate a seed-based correlation matrix,
    the default seed is located in right PCC.

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    proj_rawdata : path
        Location of BIDS-organized project rawdata
    proj_deriv : path
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories
    work_deriv : path
        Parent location for writing pipeline intermediates
    sing_afni : path
        Location of AFNI singularity file
    model_name : str
        [rest]
        Desired AFNI model, for triggering different workflows
    log_dir : path
        Output location for log files and scripts

    Returns
    -------
    triple
        [0] = dictionary of z-tranformed correlation matrices
        [1] = dictionary of anat files
        [2] = dictionary of func files

    Raises
    ------
    ValueError
        Model name/type not supported

    """
    # Validate
    if model_name != "rest":
        raise ValueError(f"Unsupported model name : {model_name}")

    # Check that session exists for participant
    subj_sess_raw = os.path.join(proj_rawdata, subj, sess)
    if not os.path.exists(subj_sess_raw):
        print(f"Directory not detected : {subj_sess_raw}\n\tSkipping.")

    # Setup output directory
    subj_work = os.path.join(work_deriv, "model_afni-rest", subj, sess, "func")
    if not os.path.exists(subj_work):
        os.makedirs(subj_work)

    # Extra pre-processing steps, generate deconvolution command
    sess_func, sess_anat = afni_pipelines.extra_preproc(
        subj, sess, subj_work, proj_deriv, sing_afni, do_rest=True
    )
    write_decon = deconvolve.WriteDecon(
        subj_work,
        proj_deriv,
        sess_func,
        sess_anat,
        sing_afni,
    )
    write_decon.build_decon(model_name)

    # Project regression matrix
    proj_reg = deconvolve.ProjectRest(
        subj, sess, subj_work, proj_deriv, sing_afni, log_dir
    )
    proj_reg.gen_xmatrix(write_decon.decon_cmd, write_decon.decon_name)
    proj_reg.anaticor(
        write_decon.decon_name,
        write_decon.epi_masked,
        sess_anat,
        sess_func,
    )

    # Seed (sanity check) and clean
    corr_dict = proj_reg.seed_corr(sess_anat)
    helper.MoveFinal(subj, sess, proj_deriv, subj_work, sess_anat, model_name)
    return (corr_dict, sess_anat, sess_func)


# %%
def afni_extract(
    proj_dir, subj_list, model_name, group_mask=True, comb_all=True
):
    """Extract sub-brick betas and generate dataframe.

    Split AFNI deconvolved files by sub-brick and then extract the
    beta-coefficient for each behavior of interest from each voxel
    and generate a dataframe.

    A binary brain mask can be generated and used to reduce the
    size of the dataframes.

    Output of group_mask and comb_all are written to:
        <proj_dir>/analyses/model_afni

    Parameters
    ----------
    proj_dir : path
        Location of project directory
    subj_list : list
        Subject IDs to include in dataframe
    model_name : str
        [univ]
        Model identifier of deconvolved file
    group_mask : bool, optional
        Whether to generate a group intersection mask and then
        find coordinates to remove from dataframe
    comb_all : bool, optional
        Combine all participand beta dataframes into an
        omnibus one

    Raises
    ------
    ValueError
        Unexpected model_name value

    """
    if model_name != "univ":
        raise ValueError("Unexpected model_name")

    # Setup orienting variables
    out_dir = os.path.join(proj_dir, "analyses/model_afni")
    proj_deriv = os.path.join(proj_dir, "data_scanner_BIDS", "derivatives")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize beta extraction
    get_betas = group.ExtractTaskBetas(proj_dir)

    # Generate mask and identify censor coordinates
    if group_mask:
        mask_path = masks.group_mask(proj_deriv, subj_list, out_dir)
        get_betas.mask_coord(mask_path)

    # Make beta dataframe for each subject
    for subj in subj_list:
        for sess in ["ses-day2", "ses-day3"]:

            # Find decon file
            subj_deriv_func = os.path.join(
                proj_deriv, "model_afni", subj, sess, "func"
            )
            decon_path = os.path.join(
                subj_deriv_func, f"decon_{model_name}_stats_REML+tlrc.HEAD"
            )
            if not os.path.exists(decon_path):
                continue

            # Identify task, make beta dataframe
            task_path = glob.glob(
                f"{subj_deriv_func}/timing_files/*_events.1D"
            )[0]
            _, _, task, _, _ = os.path.basename(task_path).split("_")
            _ = get_betas.make_func_matrix(
                subj, sess, task, model_name, decon_path
            )

    # Combine all participant dataframes
    if comb_all:
        _ = group.comb_matrices(subj_list, model_name, proj_deriv, out_dir)


# %%
def fsl_task_first(
    subj,
    sess,
    model_name,
    proj_rawdata,
    proj_deriv,
    work_deriv,
    log_dir,
):
    """Title.

    Desc.

    """
    # check model_namel

    # Check that session exists for participant
    subj_sess_raw = os.path.join(proj_rawdata, subj, sess)
    if not os.path.exists(subj_sess_raw):
        print(f"Directory not detected : {subj_sess_raw}\n\tSkipping.")

    # Setup output directory
    subj_work = os.path.join(
        work_deriv, f"model_fsl-{model_name}", subj, sess, "func"
    )
    if not os.path.exists(subj_work):
        os.makedirs(subj_work)

    # Find events files
    sess_events = sorted(glob.glob(f"{subj_sess_raw}/func/*events.tsv"))
    if not sess_events:
        raise FileNotFoundError(
            f"Expected BIDs events files in {subj_sess_raw}"
        )

    # Identify and validate task name
    _task_short = (
        os.path.basename(sess_events[0]).split("task-")[-1].split("_")[0]
    )
    task = "task-" + _task_short
    if task not in ["task-movies", "task-scenarios"]:
        raise ValueError(f"Unexpected task name : {task}")

    # Make condition files
    make_cf = fsl.ConditionFiles(subj, sess, task, subj_work, sess_events)
    for run_num in make_cf.run_list:
        make_cf.common_events(run_num)
        if model_name == "sep":
            make_cf.session_separate_events(run_num)

    # Find confounds files, extract relevant columns
    fp_subj_sess = os.path.join(
        proj_deriv, "pre_processing/fmriprep", subj, sess
    )
    sess_confounds = sorted(
        glob.glob(f"{fp_subj_sess}/func/*{task}*timeseries.tsv")
    )
    if not sess_confounds:
        raise FileNotFoundError(
            f"Expected fMRIPrep confounds files in {fp_subj_sess}"
        )
    for conf_path in sess_confounds:
        _ = fsl.confounds(conf_path, subj_work)

    #
    fd_subj_sess = os.path.join(
        proj_deriv, "pre_processing/fsl_denoise", subj, sess
    )
    sess_preproc = sorted(
        glob.glob(f"{fd_subj_sess}/func/*tfiltMasked_bold.nii.gz")
    )
    if not sess_preproc:
        raise FileNotFoundError(
            f"Expected fsl_denoise files in {fd_subj_sess}"
        )
    for preproc_path in sess_preproc:
        pass

    # # clean up
    # # TODO move to fsl method
    # cp_dir = os.path.join(work_deriv, "model_fsl-task", subj)
    # final_dir = os.path.join(proj_deriv, "model_fsl")
    # h_sp = subprocess.Popen(
    #     f"cp -r {cp_dir} {final_dir}", shell=True, stdout=subprocess.PIPE
    # )
    # _ = h_sp.communicate()
    # h_sp.wait()
    # shutil.rmtree(cp_dir)
