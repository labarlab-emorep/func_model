"""Pipelines supporting AFNI and FSL.

afni_task       : GLM task data via AFNI
afni_rest       : deprecated, GLM rest data via AFNI
afni_extract    : deprecated, extract task beta-coefficients from AFNI GLM
afni_ttest      : conduct Student's T via ETAC method
afni_mvm        : conduct ANOVA-style analyses via 3dMVM

"""

import os
import glob
import shutil
from typing import Union
from func_model.resources import preprocess
from func_model.resources import deconvolve
from func_model.resources import masks
from func_model.resources import helper
from func_model.resources import group
from func_model.workflows import wf_fsl


class _SyncData(wf_fsl._SupportFsl):
    """Coordinate setup, data download, output upload."""

    def __init__(
        self, subj: str, sess: str, work_deriv: Union[str, os.PathLike]
    ):
        """Initialize."""
        self._subj = subj
        self._sess = sess
        self._work_deriv = work_deriv
        self._keoki_path = (
            "/mnt/keoki/experiments2/EmoRep/"
            + "Exp2_Compute_Emotion/data_scanner_BIDS"
        )
        super().__init__(self._keoki_path)

    def setup_indiv(self) -> tuple:
        """Setup working directories for individual models.

        Returns
        -------
        tuple
            - [0] = work_deriv/model_afni/subj/sess/func
            - [1] = work_deriv/fmriprep/subj/sess
            - [2] = work_deriv/rawdata/subj/sess

        """
        # Make fmriprep dir, manage conflict between sessions
        self._work_fp = os.path.join(self._work_deriv, "fmriprep", self._subj)
        try:
            os.makedirs(self._work_fp)
        except FileExistsError:
            pass

        # make rawdata, work dir
        self._work_raw = os.path.join(
            self._work_deriv, "rawdata", self._subj, self._sess, "func"
        )
        self._subj_work = os.path.join(
            self._work_deriv, "model_afni", self._subj, self._sess, "func"
        )
        for _dir in [self._work_raw, self._subj_work]:
            if not os.path.exists(_dir):
                os.makedirs(_dir)

        # Return afni, fmriprep, rawdata dir paths
        return (
            self._subj_work,
            os.path.join(self._work_fp, self._sess),
            os.path.dirname(self._work_raw),
        )

    @property
    def _ls2_addr(self) -> str:
        """Return user@labarserv2."""
        return os.environ["USER"] + "@" + self._ls2_ip

    def get_fmriprep(self):
        """Download fMRIPrep for subj, sess."""
        if not hasattr(self, "_work_fp"):
            self.setup_indiv()

        # Check for fmriprep output
        chk_file = os.path.join(
            self._work_fp,
            self._sess,
            "anat",
            f"{self._subj}_{self._sess}_desc-preproc_T1w.nii.gz",
        )
        if os.path.exists(chk_file):
            return

        # Get fMRIPrep - anat, motion confs, func
        source_fp = os.path.join(
            self._keoki_path,
            "derivatives",
            "pre_processing",
            "fmriprep",
            self._subj,
            self._sess,
        )
        _, _ = self._submit_rsync(
            f"{self._ls2_addr}:{source_fp}", self._work_fp
        )

        # Verify download
        if not os.path.exists(chk_file):
            raise FileNotFoundError(
                f"fMRIPrep download failed for {self._subj}, {self._sess}"
            )

    def get_events(self) -> list:
        """Download and return list of rawdata events files for subj, sess."""
        if not hasattr(self, "_work_raw"):
            self.setup_indiv()

        # Check for existing events files
        event_list = sorted(glob.glob(f"{self._work_raw}/*events.tsv"))
        if event_list:
            return event_list

        # Get rawdata events
        source_raw = os.path.join(
            self._keoki_path,
            "rawdata",
            self._subj,
            self._sess,
            "func",
            "*events.tsv",
        )
        _, _ = self._submit_rsync(
            f"{self._ls2_addr}:{source_raw}", self._work_raw
        )

        # Verify download
        event_list = sorted(glob.glob(f"{self._work_raw}/*events.tsv"))
        if not event_list:
            raise ValueError(
                f"Rsync failure, missing events files at : {self._work_raw}"
            )
        return event_list

    def clean_deriv(self, model_name: str, sess_anat: dict):
        """Remove unneeded files from model_afni."""
        self._model_name = model_name
        self._sess_anat = sess_anat
        if not hasattr(self, "_subj_work"):
            self.setup_indiv()

        # Identify files to save, remove all others
        save_list = (
            self._save_rest() if model_name == "rest" else self._save_task()
        )
        for file_name in os.listdir(self._subj_work):
            wash_path = os.path.join(self._subj_work, file_name)
            if wash_path not in save_list:
                os.remove(wash_path)

    def _save_task(self) -> list:
        """Return list of important task decon files."""
        # Check for pipeline output
        chk_file = os.path.join(
            self._subj_work, f"decon_{self._model_name}_stats_REML+tlrc.HEAD"
        )
        if not os.path.exists(chk_file):
            raise FileNotFoundError(
                f"Missing expected output file : {chk_file}"
            )

        # Build return list
        save_list = glob.glob(f"{self._subj_work}/*decon_{self._model_name}*")
        save_list.append(self._sess_anat["mask-WMe"])
        save_list.append(self._sess_anat["mask-int"])
        save_list.append(os.path.join(self._subj_work, "motion_files"))
        save_list.append(os.path.join(self._subj_work, "timing_files"))
        return save_list

    def _save_rest(self):
        """Return list of important rest decon files."""
        stat_list = glob.glob(f"{self._subj_work}/decon_rest_anaticor*+tlrc.*")
        if not stat_list:
            raise FileNotFoundError(
                f"Missing decon_rest files in {self._subj_work}"
            )

        # Make return lists
        seed_list = glob.glob(f"{self._subj_work}/seed_*")
        x_list = glob.glob(f"{self._subj_work}/X.decon_rest.*")
        save_list = stat_list + seed_list + x_list
        save_list.append(self._sess_anat["mask-int"])
        save_list.append(f"{self._subj_work}/decon_rest.sh")
        return save_list

    def send_decon(self):
        """Send decon output to Keoki."""
        if not hasattr(self, "_subj_work"):
            self.setup_indiv()

        # Make output destination
        keoki_dst = os.path.join(
            self._keoki_path, "derivatives/model_afni", self._subj, self._sess
        )
        make_dst = f"""\
            ssh \
                -i {self._rsa_key} \
                {self._ls2_addr} \
                " command ; bash -c 'mkdir -p {keoki_dst}'"
        """
        _, _ = self._quick_sp(make_dst)

        # Send output directory to Keoki
        _, _ = self._submit_rsync(
            self._subj_work, f"{self._ls2_addr}:{keoki_dst}"
        )


def afni_task(
    subj,
    sess,
    work_deriv,
    model_name,
    log_dir,
):
    """Conduct AFNI-based deconvolution.

    Download data from Keoki, generate timing files from
    rawdata events, motion files and preprocessed files
    from fMRIPrep, then generate deconvultion files,
    nuissance files, and execute 3dREMLfit.

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    work_deriv : str, os.PathLike
        Parent location for writing pipeline intermediates
    model_name : str
        {"univ", "mixed"}
        Desired AFNI model, for triggering different workflows
    log_dir : str, os.PathLike
        Output location for log files and scripts

    """
    if model_name not in ["univ", "mixed"]:
        raise ValueError(f"Unsupported model name : {model_name}")

    # Setup, download required files
    sync_data = _SyncData(subj, sess, work_deriv)
    subj_work, subj_fp, subj_raw = sync_data.setup_indiv()
    sync_data.get_fmriprep()
    sess_events = sync_data.get_events()

    # Get and validate task name
    task = (
        "task-"
        + os.path.basename(sess_events[0]).split("task-")[-1].split("_")[0]
    )
    if not helper.valid_task(task):
        raise ValueError(f"Expected task name : {task}")

    # Extra pre-processing steps
    sess_func, sess_anat = preprocess.extra_preproc(
        subj, sess, subj_work, work_deriv
    )

    # Make AFNI-style motion and censor files
    make_motion = deconvolve.MotionCensor(
        subj_work, work_deriv, sess_func["func-motion"]
    )
    sess_func["mot-mean"] = make_motion.mean_motion()
    sess_func["mot-deriv"] = make_motion.deriv_motion()
    sess_func["mot-cens"] = make_motion.censor_volumes()
    _ = make_motion.count_motion()

    # Generate and compile timing files
    make_tf = deconvolve.TimingFiles(subj, sess, task, subj_work, sess_events)
    tf_list = make_tf.common_events()
    tf_list += make_tf.session_events()
    tf_list += make_tf.select_events()
    if model_name == "mixed":
        tf_list += make_tf.session_blocks()

    # Organize timing files by description
    sess_timing = {}
    for tf_path in tf_list:
        h_key = os.path.basename(tf_path).split("desc-")[1].split("_")[0]
        sess_timing[h_key] = tf_path

    # Generate deconvolution command
    run_reml = deconvolve.RunReml(
        subj,
        sess,
        subj_work,
        work_deriv,
        sess_anat,
        sess_func,
        log_dir,
    )
    run_reml.build_decon(model_name, sess_tfs=sess_timing)

    # Use decon command to make REMl command, execute REML
    run_reml.generate_reml()
    _ = run_reml.exec_reml()

    # Send data to Keoki, clean
    sync_data.clean_deriv(model_name, sess_anat)
    sync_data.send_decon()
    shutil.rmtree(os.path.dirname(subj_work))
    shutil.rmtree(subj_fp)
    shutil.rmtree(subj_raw)


def afni_rest(
    subj,
    sess,
    proj_rawdata,
    proj_deriv,
    work_deriv,
    model_name,
    log_dir,
):
    """Conduct AFNI-styled resting state analysis for sanity checking.

    Deprecated.

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
    return

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
    sess_func, sess_anat = preprocess.extra_preproc(
        subj, sess, subj_work, proj_deriv, do_rest=True
    )
    write_decon = deconvolve.WriteDecon(
        subj_work,
        proj_deriv,
        sess_func,
        sess_anat,
    )
    write_decon.build_decon(model_name)

    # Project regression matrix
    proj_reg = deconvolve.ProjectRest(
        subj, sess, subj_work, proj_deriv, log_dir
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


def afni_extract(
    proj_dir, subj_list, model_name, group_mask="template", comb_all=True
):
    """Extract sub-brick betas and generate dataframe.

    Deprecated.

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
    return

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
    get_betas = group.AfniExtractTaskBetas(proj_dir)

    # Generate mask and identify censor coordinates
    if group_mask == "template":
        mask_path = masks.tpl_gm(out_dir)
    elif group_mask == "intersection":
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


def afni_ttest(
    task, model_name, stat, emo_name, work_deriv, log_dir, blk_coef=False
):
    """Conduct T-tests in AFNI using the ETAC method.

    Conduct stimulus-specific (movies, scenarios) t-tests comparing
    emotion against zero (student) or washout (paired).

    Parameters
    ----------
    task : str
        {"task-movies", "task-scenarios"}
        BIDS task identifier
    model_name : str
        {"univ", "mixed"}
        Type of AFNI deconvolution
    stat : str
        {"student", "paired"}
        Type of T-test to conduct
    emo_name : str, key of afni.helper.emo_switch
        Lower case emotion name
    work_deriv : str, os.PathLike
        Parent location for writing pipeline intermediates
    log_dir : str, os.PathLike
        Location of logging files
    blk_coef : bool, optional
        Use block (blk) coefficients rather than event, only
        available when model_name=mixed.

    """
    # Validate options
    if not helper.valid_task(task):
        raise ValueError(f"Unexpected task value : {task}")
    if not helper.valid_univ_test(stat):
        raise ValueError(f"Unexpected stat test name : {stat}")
    emo_switch = helper.emo_switch()
    if emo_name not in emo_switch.keys():
        raise ValueError(f"Unexpected emotion name : {emo_name}")
    if blk_coef and model_name != "mixed":
        raise ValueError(
            "Option blk_coef=True only available for model_name=mixed"
        )

    # Setup
    sync_data = helper.SyncGroup(work_deriv)
    model_indiv, model_group = sync_data.setup_group()

    # Identify participant/sessions with desired task from timing file
    search_path = os.path.join(
        model_indiv, "sub-*", "ses-*", "func", "timing_files"
    )
    wash_list = sorted(
        glob.glob(f"{search_path}/*_{task}_desc-comWas_events.1D")
    )
    if not wash_list:
        raise ValueError("Failed to detect desc-comWas timing files.")

    # Build dict needed by group.EtacTest.write_exec
    decon_dict = {}
    for wash_path in wash_list:

        # Find the decon file
        decon_path = os.path.join(
            os.path.dirname(os.path.dirname(wash_path)),
            f"decon_{model_name}_stats_REML+tlrc.HEAD",
        )
        if not os.path.exists(decon_path):
            raise FileNotFoundError(f"Expected decon file : {decon_path}")

        # Identify subj, sess info
        subj, _sess, _task, _desc, _suff = os.path.basename(wash_path).split(
            "_"
        )
        decon_dict[subj] = decon_path

    # Make, get mask
    mask_path = masks.tpl_gm(model_group)

    # Build coefficient name to match sub-brick, recreate name
    # specified by resources.deconvolve.TimingFiles.session_events,
    # and append AFNI coefficient title.
    task_short = task.split("-")[1][:3]
    if task_short not in ["mov", "sce"]:
        raise ValueError("Problem splitting task name")
    emo_short = emo_switch[emo_name]
    sub_label = (
        f"blk{task_short.title()[0]}{emo_short}#0_Coef"
        if blk_coef
        else f"{task_short}{emo_short}#0_Coef"
    )

    # Generate, execute ETAC command
    run_etac = group.EtacTest(model_group, mask_path)
    out_path = run_etac.write_exec(
        task,
        model_name,
        stat,
        emo_name,
        decon_dict,
        sub_label,
        log_dir,
        blk_coef,
    )

    # Send output to Keoki, clean
    out_dir = os.path.dirname(out_path)
    sync_data.send_etac(out_dir)
    shutil.rmtree(out_dir)


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
    if not helper.valid_mvm_test(model_name):
        raise ValueError(f"Unexpected model name : {model_name}")
    emo_switch = helper.emo_switch()
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
    decon_dict = {}
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
        decon_dict[subj] = {}

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
            decon_dict[subj][task] = {
                "sess": sess,
                "decon_path": decon_path,
                "emo_label": emo_label,
                "wash_label": wash_label,
            }

    # Make, get mask
    print("\tGetting group mask")
    mask_path = masks.tpl_gm(group_dir)

    # Generate, execute ETAC command
    run_mvm = group.MvmTest(proj_dir, out_dir, mask_path)
    run_mvm.clustsim()
    _ = run_mvm.write_exec(decon_dict, model_name, emo_short)
