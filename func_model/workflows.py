"""Pipelines supporting AFNI and FSL."""
# %%
import os
import glob
from pathlib import Path
from func_model.resources import afni, fsl


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
    # Validate, check session, and setup
    if model_name not in ["univ", "mixed"]:
        raise ValueError(f"Unsupported model name : {model_name}")
    subj_sess_raw = os.path.join(proj_rawdata, subj, sess)
    if not os.path.exists(subj_sess_raw):
        print(f"Directory not detected : {subj_sess_raw}\n\tSkipping.")
        return
    subj_work = os.path.join(work_deriv, "model_afni-task", subj, sess, "func")
    if not os.path.exists(subj_work):
        os.makedirs(subj_work)

    # Extra pre-processing steps
    sess_func, sess_anat = afni.preprocess.extra_preproc(
        subj, sess, subj_work, proj_deriv, sing_afni
    )

    # Find events files, get and validate task name
    sess_events = sorted(glob.glob(f"{subj_sess_raw}/func/*events.tsv"))
    if not sess_events:
        raise FileNotFoundError(
            f"Expected BIDs events files in {subj_sess_raw}"
        )
    task = os.path.basename(sess_events[0]).split("task-")[-1].split("_")[0]
    if not afni.helper.valid_task(f"task-{task}"):
        raise ValueError(f"Expected task names movies|scenarios, found {task}")

    # Generate, organize timing files
    make_tf = afni.deconvolve.TimingFiles(subj_work, sess_events)
    tf_com = make_tf.common_events(subj, sess, task)
    tf_sess = make_tf.session_events(subj, sess, task)
    tf_sel = make_tf.select_events(subj, sess, task)
    tf_all = tf_com + tf_sess + tf_sel
    if model_name == "mixed":
        tf_blk = make_tf.session_blocks(subj, sess, task)
        tf_all = tf_com + tf_sess + tf_sel + tf_blk

    sess_timing = {}
    for tf_path in tf_all:
        h_key = os.path.basename(tf_path).split("desc-")[1].split("_")[0]
        sess_timing[h_key] = tf_path

    # Generate deconvolution command
    write_decon = afni.deconvolve.WriteDecon(
        subj_work,
        proj_deriv,
        sess_func,
        sess_anat,
        sing_afni,
    )
    write_decon.build_decon(model_name, sess_tfs=sess_timing)

    # Use decon command to make REMl command, execute REML
    make_reml = afni.deconvolve.RunReml(
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
    afni.helper.MoveFinal(
        subj, sess, proj_deriv, subj_work, sess_anat, model_name
    )
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
    # Validate and check session, setup
    if model_name != "rest":
        raise ValueError(f"Unsupported model name : {model_name}")
    subj_sess_raw = os.path.join(proj_rawdata, subj, sess)
    if not os.path.exists(subj_sess_raw):
        print(f"Directory not detected : {subj_sess_raw}\n\tSkipping.")
        return
    subj_work = os.path.join(work_deriv, "model_afni-rest", subj, sess, "func")
    if not os.path.exists(subj_work):
        os.makedirs(subj_work)

    # Extra pre-processing steps, generate deconvolution command
    sess_func, sess_anat = afni.preprocess.extra_preproc(
        subj, sess, subj_work, proj_deriv, sing_afni, do_rest=True
    )
    write_decon = afni.deconvolve.WriteDecon(
        subj_work,
        proj_deriv,
        sess_func,
        sess_anat,
        sing_afni,
    )
    write_decon.build_decon(model_name)

    # Project regression matrix
    proj_reg = afni.deconvolve.ProjectRest(
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
    afni.helper.MoveFinal(
        subj, sess, proj_deriv, subj_work, sess_anat, model_name
    )
    return (corr_dict, sess_anat, sess_func)


# %%
def afni_extract(
    proj_dir, subj_list, model_name, group_mask="template", comb_all=True
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
    group_mask : str, optional
        [template | intersection]
        Generate a group-level mask, used to identify and remove
        voxels of no interest from beta dataframe
    comb_all : bool, optional
        Combine all participant beta dataframes into an
        omnibus one

    Raises
    ------
    ValueError
        Unexpected model_name value
        Unexpected group_mask value

    """
    # Validate and setup
    if model_name != "univ":
        raise ValueError("Unexpected model_name")
    if group_mask not in ["template", "intersection"]:
        raise ValueError("unexpected group_mask parameter")
    out_dir = os.path.join(proj_dir, "analyses/model_afni")
    proj_deriv = os.path.join(proj_dir, "data_scanner_BIDS", "derivatives")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize beta extraction
    get_betas = afni.group.ExtractTaskBetas(proj_dir)

    # Generate mask and identify censor coordinates
    if group_mask == "template":
        mask_path = afni.masks.tpl_gm(out_dir)
    elif group_mask == "intersection":
        mask_path = afni.masks.group_mask(proj_deriv, subj_list, out_dir)
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
        _ = afni.group.comb_matrices(
            subj_list, model_name, proj_deriv, out_dir
        )


# %%
def afni_ttest(task, model_name, emo_name, proj_dir):
    """Conduct T-tests in AFNI using the ETAC method.

    Conduct stimulus-specific (movies, scenarios) t-tests comparing
    emotion against zero (student) or washout (paired). Output
    scripts and files are written to:
        <proj_dir>/analyses/model_afni/ttest_<model_name>/<task>_<emo_name>

    Parameters
    ----------
    task : str
        [task-movies | task-scenarios]
        BIDS task identifier
    model_name : str
        [student | paired]
        Type of T-test to conduct
    emo_name : str, key of afni.helper.emo_switch
        Lower case emotion name
    proj_dir : path
        Project directory location, should contain
        data_scanner_BIDS/derivatives/model_afni

    Raises
    ------
    FileNotFoundError
        Missing required directory
    ValueError
        Unexpected argument parameter

    """
    # Validate paths
    if not os.path.exists(proj_dir):
        raise FileNotFoundError(
            f"Missing expected project directory : {proj_dir}"
        )
    afni_deriv = os.path.join(
        proj_dir, "data_scanner_BIDS/derivatives", "model_afni"
    )
    if not os.path.exists(afni_deriv):
        raise FileNotFoundError(f"Missing expected directory : {afni_deriv}")

    # Validate strings
    if not afni.helper.valid_task(task):
        raise ValueError(f"Unexpected task value : {task}")
    if not afni.helper.valid_univ_test(model_name):
        raise ValueError(f"Unexpected model name : {model_name}")
    emo_switch = afni.helper.emo_switch()
    if emo_name not in emo_switch.keys():
        raise ValueError(f"Unexpected emotion name : {emo_name}")

    # Setup
    print(f"\nConducting {model_name} ETAC for {emo_name}")
    group_dir = os.path.join(proj_dir, "analyses/model_afni")
    out_dir = os.path.join(
        group_dir, f"ttest_{model_name}", f"{task}_{emo_name}"
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Identify participant/sessions with desired task from timing file
    task_subj = sorted(
        glob.glob(
            f"{afni_deriv}/**/*_{task}_desc-comWas_events.1D",
            recursive=True,
        )
    )
    if not task_subj:
        raise ValueError("Failed to detect desc-comWas timing files.")

    # Build dict needed by group.EtacTest.write_exec
    group_dict = {}
    for file_path in task_subj:
        decon_path = os.path.join(
            Path(file_path).parents[1], "decon_univ_stats_REML+tlrc.HEAD"
        )
        if os.path.exists(decon_path):
            subj, sess, _task, _desc, _suff = os.path.basename(
                file_path
            ).split("_")
            group_dict[subj] = {
                "sess": sess,
                "decon_path": decon_path,
            }

    # Make, get mask
    mask_path = afni.masks.tpl_gm(group_dir)

    # Build coefficient name to match sub-brick, recreate name
    # specified by afni.deconvolve.TimingFiles.session_events,
    # and append AFNI coefficient title.
    task_short = task.split("-")[1][:3]
    if task_short not in ["mov", "sce"]:
        raise ValueError("Problem splitting task name")
    emo_short = emo_switch[emo_name]
    sub_label = task_short + emo_short + "#0_Coef"

    # Generate, execute ETAC command
    run_etac = afni.group.EtacTest(proj_dir, out_dir, mask_path)
    _ = run_etac.write_exec(model_name, emo_short, group_dict, sub_label)


# %%
def afni_mvm(proj_dir, model_name, emo_name):
    """Conduct ANOVA-style tests in AFNI via 3dMVM.

    Identify participants with deconvolved files from both sessions
    and then conduct an ANOVA-styled analysis for the specified emotion.
    Cluster simulations (Monte Carlo) are written to the same directory
    as the group-level mask, and 3dMVM scripts and output are written to:
        <proj_dir>/analyses/model_afni/mvm_<model_name>/<emo_name>_vs_washout

    Currently, only a two-factor repeated-measures ANOVA analysis is
    supported, with factor A as session stimulus (movies, scenarios), and
    factor B as stimulus type (emotion, washout). Main, interactive effects
    are generated as well as an emotion-washout T-stat.

    Parameters
    ----------
    proj_dir : path
        Project directory location, should contain
        data_scanner_BIDS/derivatives/model_afni
    model_name : str
        [rm]
        Model identifier
    emo_name : str, key of afni.helper.emo_switch
        Lower case emotion name

    Raises
    ------
    FileNotFoundError
        Missing expected directories
    ValueError
        Unexpected model_name, emo_name
        Unexpected first three characters of task name

    """
    # Validate paths
    if not os.path.exists(proj_dir):
        raise FileNotFoundError(
            f"Missing expected project directory : {proj_dir}"
        )
    afni_deriv = os.path.join(
        proj_dir, "data_scanner_BIDS/derivatives", "model_afni"
    )
    if not os.path.exists(afni_deriv):
        raise FileNotFoundError(f"Missing expected directory : {afni_deriv}")

    # Validate strings
    if not afni.helper.valid_mvm_test(model_name):
        raise ValueError(f"Unexpected model name : {model_name}")
    emo_switch = afni.helper.emo_switch()
    if emo_name not in emo_switch.keys():
        raise ValueError(f"Unexpected emotion name : {emo_name}")

    # Setup output location
    print(f"\nConducting {model_name} MVM for {emo_name}")
    group_dir = os.path.join(proj_dir, "analyses/model_afni")
    out_dir = os.path.join(
        group_dir, f"mvm_{model_name}", f"{emo_name}_vs_washout"
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get AFNI emo name and washout subbrick label
    emo_short = emo_switch[emo_name]
    wash_label = "comWas#0_Coef"

    # Make group dictionary required by group.MvmTest.write_exec
    print("\tBuilding group dictionary")
    subj_all = [
        os.path.basename(x) for x in sorted(glob.glob(f"{afni_deriv}/sub-*"))
    ]
    group_dict = {}
    for subj in subj_all:

        # Only include participants with deconvolved data from both sessions
        decon_list = sorted(
            glob.glob(
                f"{afni_deriv}/{subj}/**/decon_univ_stats_REML+tlrc.HEAD",
                recursive=True,
            )
        )
        if len(decon_list) != 2:
            print(
                "Unexpected number of deconvolved files, excluding "
                + f"{subj} from group-level model"
            )
            continue
        group_dict[subj] = {}

        # Identify the task name from washout timing file
        for decon_path in decon_list:
            search_path = os.path.join(
                os.path.dirname(decon_path), "timing_files"
            )
            wash_path = glob.glob(f"{search_path}/*_desc-comWas_events.1D")[0]
            wash_file = os.path.basename(wash_path)
            _subj, sess, task, _desc, _suff = wash_file.split("_")

            # Reconstruct subbrick label for the emotion
            task_short = task.split("-")[1][:3]
            if task_short not in ["mov", "sce"]:
                raise ValueError("Problem splitting task name")
            emo_label = task_short + emo_short + "#0_Coef"

            # Fill dictionary with required structure, items
            group_dict[subj][task] = {
                "sess": sess,
                "decon_path": decon_path,
                "emo_label": emo_label,
                "wash_label": wash_label,
            }

    # Make, get mask
    print("\tGetting group mask")
    mask_path = afni.masks.tpl_gm(group_dir)

    # Generate, execute ETAC command
    run_mvm = afni.group.MvmTest(proj_dir, out_dir, mask_path)
    run_mvm.clustsim()
    _ = run_mvm.write_exec(group_dict, model_name, emo_short)


# %%
def fsl_task_first(
    subj,
    sess,
    model_name,
    model_level,
    proj_rawdata,
    proj_deriv,
    work_deriv,
    log_dir,
):
    """Run an FSL first-level model for task EPI data.

    Generate required confounds, condition, and design files and then
    use FSL's FEAT to run a first-level model.

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    model_name : str
        Name of FSL model, for keeping condition files and
        output organized
    model_level : str
        [first]
        Level of FSL model
    proj_rawdata : path
        Location of BIDS rawdata
    proj_deriv : path
        Location of project BIDs derivatives, for finding
        preprocessed output
    work_deriv : path
        Output location for intermediates
    log_dir : path
        Output location for log files and scripts

    Raises
    ------
    ValueError
        Unexpected parameter values
        Unexpected task name

    """
    # Check arguments, that data exist
    if not fsl.helper.valid_name(model_name):
        raise ValueError(f"Unexpected model name : {model_name}")
    if model_level != "first":
        raise ValueError(f"Unexpected model level : {model_level}")
    chk_sess = os.path.join(proj_rawdata, subj, sess)
    if not os.path.exists(chk_sess):
        print(f"Directory not detected : {chk_sess}\n\tSkipping.")
        return

    # Setup output locations
    subj_work = os.path.join(
        work_deriv, f"model_fsl-{model_name}", subj, sess, "func"
    )
    subj_final = os.path.join(proj_deriv, "model_fsl", subj, sess)
    for _dir in [subj_work, subj_final]:
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    # Determine and check task name
    search_path = os.path.join(proj_rawdata, subj, sess, "func")
    event_path = glob.glob(f"{search_path}/*events.tsv")[0]
    event_file = os.path.basename(event_path)
    task = "task-" + event_file.split("task-")[-1].split("_")[0]
    if not fsl.helper.valid_task(task):
        raise ValueError(f"Unexpected task name : {task}")

    # Make condition, confound, and design files
    fsl.wrap.make_condition_files(
        subj, sess, task, model_name, subj_work, proj_rawdata
    )
    fsl.wrap.make_confound_files(subj, sess, task, subj_work, proj_deriv)
    fsf_list = fsl.wrap.write_first_fsf(
        subj, sess, task, model_name, subj_work, proj_deriv
    )

    # Execute each run's model
    for fsf_path in fsf_list:
        _ = fsl.model.run_feat(
            fsf_path, subj, sess, model_name, model_level, log_dir
        )
    fsl.helper.clean_up(subj_work, subj_final)


# %%
def fsl_extract(
    proj_dir,
    subj_list,
    model_name,
    model_level,
    group_mask="template",
    comb_all=True,
):
    """Title.

    Parameters
    ----------

    """
    # check model_name, model_level

    #
    if group_mask not in ["template", "intersection"]:
        raise ValueError("unexpected group_mask parameter")

    #
    out_dir = os.path.join(proj_dir, "analyses/model_fsl")
    proj_deriv = os.path.join(proj_dir, "data_scanner_BIDS", "derivatives")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize beta extraction
    get_betas = fsl.group.ExtractTaskBetas(proj_dir)

    # Generate mask and identify censor coordinates
    if group_mask == "template":
        mask_path = afni.masks.tpl_gm(out_dir)
    elif group_mask == "intersection":
        mask_path = afni.masks.group_mask(proj_deriv, subj_list, out_dir)
    get_betas.mask_coord(mask_path)

    # Make beta dataframe for each subject
    for subj in subj_list:
        for sess in ["ses-day2", "ses-day3"]:

            #
            subj_deriv_func = os.path.join(
                proj_deriv, "model_fsl", subj, sess, "func"
            )
            task_path = glob.glob(
                f"{subj_deriv_func}/condition_files/*_events.txt"
            )[0]
            _subj, _sess, task, _run, _desc, _suff = os.path.basename(
                task_path
            ).split("_")
            design_list = sorted(
                glob.glob(f"{subj_deriv_func}/run-*/design.con")
            )
            get_betas.make_func_matrix(
                subj,
                sess,
                task,
                model_name,
                model_level,
                design_list,
                subj_deriv_func,
            )

    # Combine all participant dataframes
    if comb_all:
        _ = fsl.group.comb_matrices(
            subj_list, model_name, model_level, proj_deriv, out_dir
        )
