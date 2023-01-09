"""Pipelines supporting AFNI and FSL."""
# %%
import os
from func_model import run_pipeline
from func_model import afni


# %%
def pipeline_afni_task(
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
    Supplies high-level steps, coordinates actual work in run_pipeline
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
        [univ | indiv]
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
    if model_name not in ["univ", "indiv"]:
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
    sess_func, sess_anat = run_pipeline.afni_preproc(
        subj, sess, subj_work, proj_deriv, sing_afni
    )

    # Generate timing files - find appropriate pipeline for model_name
    pipe_mod = __import__(
        "func_model.run_pipeline", fromlist=[f"afni_{model_name}_tfs"]
    )
    tf_pipe = getattr(pipe_mod, f"afni_{model_name}_tfs")
    sess_timing = tf_pipe(subj, sess, subj_work, subj_sess_raw)

    # Generate deconvolution command
    write_decon = afni.WriteDecon(
        subj_work,
        proj_deriv,
        sess_func,
        sess_anat,
        sing_afni,
    )
    write_decon.build_decon(model_name, sess_tfs=sess_timing)

    # Use decon command to make REMl command, execute REML
    make_reml = afni.RunReml(
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
    wf_done = afni.move_final(
        subj, sess, proj_deriv, subj_work, sess_anat, model_name
    )
    if wf_done:
        return (sess_timing, sess_anat, sess_func)


def pipeline_afni_rest(
    subj,
    sess,
    proj_rawdata,
    proj_deriv,
    work_deriv,
    sing_afni,
    model_name,
    log_dir,
):
    """Title.

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

    # Extra pre-processing steps
    sess_func, sess_anat = run_pipeline.afni_preproc(
        subj, sess, subj_work, proj_deriv, sing_afni, do_rest=True
    )

    # Generate deconvolution command
    write_decon = afni.WriteDecon(
        subj_work,
        proj_deriv,
        sess_func,
        sess_anat,
        sing_afni,
    )
    # decon_cmd, decon_name, epi_mask = write_decon.build_decon(model_name)
    write_decon.build_decon(model_name)

    # Project regression matrix
    proj_reg = afni.ProjectRest(
        subj, sess, subj_work, proj_deriv, sing_afni, log_dir
    )
    xmat_path = proj_reg.gen_xmatrix(
        write_decon.decon_cmd, write_decon.decon_name
    )
    sess_func["func-proj"] = proj_reg.anaticor(
        write_decon.decon_name,
        write_decon.epi_mask,
        xmat_path,
        sess_anat,
        sess_func,
    )

    # Seed - sanity check
    corr_dict = proj_reg.seed_corr(sess_anat)

    # # Clean
    # wf_done = afni.move_final(
    #     subj, sess, proj_deriv, subj_work, sess_anat, model_name
    # )
    # if wf_done:
    #     return (sess_tfs, sess_anat, sess_func)


def pipeline_fsl():
    """Title.

    Desc.

    """
    pass
