"""Pipelines supporting AFNI and FSL."""
# %%
import os
from func_model import run_pipeline
from func_model import afni


# %%
def pipeline_afni(
    subj, sess, proj_rawdata, proj_deriv, work_deriv, sing_afni, log_dir
):
    """Title.

    Desc.

    Sanity check - model processing during movie|scenario presenation.

    Parameters
    ----------
    subj
    sess
    proj_rawdata
    proj_deriv
    work_deriv
    sing_afni
    log_dir

    Raises
    ------
    FileNotFoundError
    ValueError

    """
    # Check that session exists for participant
    subj_sess_raw = os.path.join(proj_rawdata, subj, sess)
    if not os.path.exists(subj_sess_raw):
        print(f"Directory not detected : {subj_sess_raw}\n\tSkipping.")

    # Setup output directory
    subj_work = os.path.join(work_deriv, "model_afni", subj, sess, "func")
    if not os.path.exists(subj_work):
        os.makedirs(subj_work)

    # Extra pre-processing steps
    sess_func, sess_anat = run_pipeline.afni_sanity_preproc(
        subj, sess, subj_work, proj_deriv, sing_afni
    )

    # Generate timing files
    sess_tfs = run_pipeline.afni_sanity_tfs(
        subj, sess, subj_work, subj_sess_raw
    )

    # Generate deconvolution matrics, REML command
    write_decon = afni.WriteDecon(
        subj,
        subj_work,
        proj_deriv,
        sess_func,
        sess_anat,
        sess_tfs,
        sing_afni,
    )
    write_decon.write_decon_sanity()
    reml_path = write_decon.generate_reml(log_dir)

    # Run REML
    sess_func["func-decon"] = afni.RunDecon(
        subj,
        subj_work,
        proj_deriv,
        reml_path,
        sess_anat,
        sess_func,
        sing_afni,
        log_dir,
    )


def pipeline_fsl():
    """Title.

    Desc.

    """
    pass
