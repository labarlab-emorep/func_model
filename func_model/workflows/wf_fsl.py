"""Pipelines supporting AFNI and FSL.

FslFirst : GLM task, rest data via FSL
FslSecond : second-level GLM of task data via FSl
FslThirdSubj : generate needed info for third-level analyses which
                collapses across subject
fsl_extract : extract task beta-coefficiens from FSL GLM
fsl_classify_mask : generate template mask from classifier output

"""
# %%
import os
import glob
import shutil
from typing import Union, Tuple
import subprocess
import pandas as pd
from multiprocessing import Process, Pool
from func_model.resources import afni
from func_model.resources import fsl


# %%
class _SupportFsl:
    """General helper methods for first- and second-level workflows."""

    def __init__(self, keoki_path: Union[str, os.PathLike]):
        """Initialize."""
        self._keoki_ip = "ccn-labarserv2.vm.duke.edu"
        self._keoki_path = keoki_path

    def _submit_rsync(self, src: str, dst: str) -> Tuple:
        """Execute rsync between DCC and labarserv2."""
        bash_cmd = f"""\
            rsync \
            -e "ssh -i {self._rsa_key}" \
            -rauv {src} {dst}
        """
        h_out, h_err = self._quick_sp(bash_cmd)
        return (h_out, h_err)

    def _quick_sp(self, bash_cmd: str) -> Tuple:
        """Spawn quick subprocess."""
        h_sp = subprocess.Popen(bash_cmd, shell=True, stdout=subprocess.PIPE)
        h_out, h_err = h_sp.communicate()
        h_sp.wait()
        return (h_out, h_err)

    def _push_data(self):
        """Make remote destination and send data there."""
        keoki_dst = os.path.join(
            self._keoki_path, "derivatives", self._final_dir, self._subj
        )
        make_dst = f"""\
            ssh \
                -i {self._rsa_key} \
                {self._user_name}@{self._keoki_ip} \
                " command ; bash -c 'mkdir -p {keoki_dst}'"
            """
        _, _ = self._quick_sp(make_dst)

        # Send data
        dst = os.path.join(
            self._keoki_proj, "derivatives", self._final_dir, self._subj
        )
        _, _ = self._submit_rsync(self._subj_final, dst)


# %%
class _SupportFslFirst(_SupportFsl):
    """Offload helper methods from FslFirst.

    Inherits _SupportFsl.

    """

    def _setup(self):
        """Set needed attrs, download and organize data."""
        self._subj_work = os.path.join(
            self._work_deriv,
            f"model_fsl-{self._model_name}",
            self._subj,
            self._sess,
            "func",
        )
        self._final_dir = (
            "model_fsl-lss" if self._model_name == "lss" else "model_fsl"
        )
        self._subj_final = os.path.join(
            self._proj_deriv, self._final_dir, self._subj, self._sess
        )
        self._subj_raw = os.path.join(
            self._proj_rawdata, self._subj, self._sess, "func"
        )
        self._subj_fp = os.path.join(
            self._proj_deriv,
            "pre_processing",
            "fmriprep",
            self._subj,
            self._sess,
            "func",
        )
        self._subj_fsl = os.path.join(
            self._proj_deriv,
            "pre_processing",
            "fsl_denoise",
            self._subj,
            self._sess,
            "func",
        )

        # Make needed directories
        for _dir in [
            self._subj_work,
            self._subj_final,
            self._subj_raw,
            self._subj_fp,
            self._subj_fsl,
        ]:
            if not os.path.exists(_dir):
                os.makedirs(_dir)

        # Download and find required data
        self._pull_data()
        self._get_preproc()

    def _pull_data(self):
        """Download required files for modeling."""
        # Set source paths and files for rest and task models
        source_time = os.path.join(
            self._keoki_proj,
            "derivatives",
            "pre_processing",
            "fmriprep",
            self._subj,
            self._sess,
            "func",
            "*timeseries.tsv",
        )
        source_preproc = os.path.join(
            self._keoki_proj,
            "derivatives",
            "pre_processing",
            "fsl_denoise",
            self._subj,
            self._sess,
            "func",
            f"*desc-{self._preproc_type}_bold.nii.gz",
        )

        # Download source
        dl_dict = {
            source_time: self._subj_fp,
            source_preproc: self._subj_fsl,
        }
        for src, dst in dl_dict.items():
            std_out, std_err = self._submit_rsync(src, dst)
            if not glob.glob(f"{dst}/{self._subj}*"):
                raise FileNotFoundError(
                    f"Missing required files at : {dst}"
                    + f"\nstdout:\n\t{std_out}\nstderr:\n\t{std_err}"
                )

        # Get events for task
        if self._model_name == "rest":
            return
        source_events = os.path.join(
            self._keoki_proj,
            "rawdata",
            self._subj,
            self._sess,
            "func",
            "*events.tsv",
        )
        std_out, std_err = self._submit_rsync(source_events, self._subj_raw)
        if not glob.glob(f"{self._subj_raw}/{self._subj}*"):
            raise FileNotFoundError(
                f"Missing required events files at : {self._subj_raw}"
                + f"\nstdout:\n\t{std_out}\nstderr:\n\t{std_err}"
            )

    def _get_preproc(self):
        """Get preprocessed EPI paths of specific task."""
        all_preproc = sorted(
            glob.glob(f"{self._subj_fsl}/*{self._preproc_type}_bold.nii.gz")
        )
        if not all_preproc:
            raise FileNotFoundError(
                f"Expected {self._preproc_type} files in {self._subj_fsl}"
            )

        if self._model_name == "rest":
            self._sess_preproc = [x for x in all_preproc if "rest" in x]
        else:
            self._sess_preproc = [x for x in all_preproc if "rest" not in x]
        if not self._sess_preproc:
            raise FileNotFoundError(
                f"Expected {self._model_name} {self._preproc_type} "
                + f"files in {self._subj_fsl}"
            )

        self._task = (
            "task-"
            + os.path.basename(self._sess_preproc[0])
            .split("task-")[1]
            .split("_")[0]
        )
        if not fsl.helper.valid_task(self._task):
            raise ValueError(f"Unexpected task name : {self._task}")

    def _get_run(self, file_name: str) -> str:
        "Set _run attr from preprocessed EPI filename."
        try:
            (
                _sub,
                _ses,
                _task,
                self._run,
                _space,
                _res,
                _desc,
                _suff,
            ) = file_name.split("_")
        except IndexError:
            raise ValueError(
                "Improperly formatted file name for preprocessed BOLD."
            )

    def _clean_dcc(self):
        """Remove rawdata, derivatives from group location."""
        rm_raw = os.path.dirname(self._subj_raw)
        rm_fp = os.path.dirname(self._subj_fp)
        rm_fsl = os.path.dirname(self._subj_fsl)
        for rm_dir in [rm_raw, rm_fp, rm_fsl, self._subj_final]:
            if os.path.exists(rm_dir):
                shutil.rmtree(rm_dir)


class _SupportFslSecond(_SupportFsl):
    """Offload helper methods from FslSecond.

    Inherits _SupportFsl.

    """

    def _setup(self):
        """Coordinate setup for design generation.

        Set required attrs, build directory trees,
        download required data, and bypass registration.

        """
        # Set reqd attrs and make directory trees
        self._subj_work = os.path.join(
            self._work_deriv,
            f"model_fsl-{self._model_name}",
            self._subj,
            self._sess,
            "func",
        )
        self._final_dir = "model_fsl"
        self._subj_deriv = os.path.join(
            self._proj_deriv, "model_fsl", self._subj, self._sess, "func"
        )
        self._subj_final = os.path.dirname(self._subj_deriv)
        for _dir in [self._subj_work, self._subj_deriv]:
            if not os.path.exists(_dir):
                os.makedirs(_dir)

        # Download data and avoid registration problem
        feat_list = self._get_first_model()
        self._bypass_reg(feat_list)

    def _get_first_model(self) -> list:
        """Download required output of FslFirst, return feat dir paths."""
        source_model = os.path.join(
            self._keoki_proj,
            "derivatives/model_fsl",
            self._subj,
            self._sess,
            "func",
            f"run*_level-first_name-{self._model_name}.feat",
        )
        std_out, std_err = self._submit_rsync(source_model, self._subj_deriv)
        run_feat = glob.glob(f"{self._subj_deriv}/run*.feat")
        if not run_feat:
            raise FileNotFoundError(
                f"Missing required files at : {self._subj_deriv}"
                + f"\nstdout:\n\t{std_out}\nstderr:\n\t{std_err}"
            )
        return run_feat

    def _bypass_reg(self, feat_list: list):
        """Avoid registration issue by supplying req'd files."""
        # Find identity file
        fsl_dir = os.environ["FSLDIR"]
        ident_path = os.path.join(fsl_dir, "etc", "flirtsch", "ident.mat")
        if not os.path.exists(ident_path):
            raise FileNotFoundError(
                f"Missing required FSL file : {ident_path}"
            )

        # Setup reg directory
        for feat_path in feat_list:
            reg_path = os.path.join(feat_path, "reg")
            if not os.path.exists(reg_path):
                os.makedirs(reg_path)

            # Get ident matrix
            ident_out = os.path.join(reg_path, "example_func2standard.mat")
            if not os.path.exists(ident_out):
                shutil.copy2(ident_path, ident_out)

            # Get ref epi
            mean_path = os.path.join(feat_path, "mean_func.nii.gz")
            mean_out = os.path.join(reg_path, "standard.nii.gz")
            if not os.path.exists(mean_out):
                shutil.copy2(mean_path, mean_out)

    def _clean_dcc(self):
        """Remove derivatives from group location."""
        shutil.rmtree(self._subj_final)


# %%
class FslFirst(_SupportFslFirst):
    """Conduct first-level models of EPI data.

    Inherits _SupportFslFirst.

    Coordinate the generation of confound and design files, and
    condition files for task-based models, then model the data
    via FSL's feat.

    This workflow will sync with Keoki (via labarserv2), pulling
    required files and uploading workflow output.

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    model_name : str
        Name of FSL model, for keeping condition files and
        output organized
    preproc_type : str
        [smoothed | scaled]
        Select preprocessed EPI file to model
    proj_rawdata : path
        Location of BIDS rawdata
    proj_deriv : path
        Location of project BIDs derivatives, for finding
        preprocessed output
    work_deriv : path
        Output location for intermediates
    log_dir : path
        Output location for log files and scripts
    user_name : str
        User name for DCC, labarserv2
    rsa_key : str, os.PathLike
        Location of RSA key for labarserv2
    keoki_path : str, os.PathLike, optional
        Location of project directory on Keoki

    Methods
    -------
    model_rest()
        Generate confounds and design files, conduct first-level
        GLM via feat
    model_task()
        Generate confounds, condition, and design files, conduct
        first-level GLM via feat

    Example
    -------
    wf_obj = wf_fsl.FslFirst(*args)
    wf_obj.model_rest()
    wf_obj.model_task()

    """

    def __init__(
        self,
        subj,
        sess,
        model_name,
        preproc_type,
        proj_rawdata,
        proj_deriv,
        work_deriv,
        log_dir,
        user_name,
        rsa_key,
        keoki_path="/mnt/keoki/experiments2/EmoRep/Exp2_Compute_Emotion/data_scanner_BIDS",  # noqa: E501
    ):
        """Initialize."""
        if not fsl.helper.valid_name(model_name):
            raise ValueError(f"Unexpected model name : {model_name}")
        if not fsl.helper.valid_preproc(preproc_type):
            raise ValueError(f"Unspported preproc type : {preproc_type}")

        print("Initializing FslFirst")
        super().__init__(keoki_path)
        self._subj = subj
        self._sess = sess
        self._model_name = model_name
        self._preproc_type = preproc_type
        self._proj_rawdata = proj_rawdata
        self._proj_deriv = proj_deriv
        self._work_deriv = work_deriv
        self._log_dir = log_dir
        self._user_name = user_name
        self._keoki_proj = (
            f"{self._user_name}@{self._keoki_ip}:{self._keoki_path}"
        )
        self._rsa_key = rsa_key

    def model_rest(self):
        """Run an FSL first-level model for resting EPI data.

        Generate required a confounds and design file, then
        use FSL's FEAT to run a first-level model.

        """
        # Setup and get required data
        print("\tRunning first-level rest model")
        self._setup()

        # Initialize needed classes, find preprocessed resting EPI
        make_fsf = fsl.model.MakeFirstFsf(
            self._subj_work, self._proj_deriv, self._model_name
        )
        rest_preproc = self._sess_preproc[0]

        # Set run and make confound file
        self._run = "run-01"
        self._make_conf()
        if not self._conf_path:
            return

        # Write and execute design.fsf
        rest_design = make_fsf.write_rest_fsf(
            self._run,
            rest_preproc,
            self._conf_path,
        )
        self._run_feat([rest_design])
        self._push_data()
        self._clean_dcc()

    def model_task(self):
        """Run an FSL first-level model for task EPI data.

        Generate required confounds, condition, and design files and then
        use FSL's FEAT to run a first-level model.

        """
        # Setup and initialize needed classes
        print("\tRunning first-level task model")
        self._setup()
        make_fsf = fsl.model.MakeFirstFsf(
            self._subj_work, self._proj_deriv, self._model_name
        )
        self._make_cf = fsl.model.ConditionFiles(
            self._subj, self._sess, self._task, self._subj_work
        )

        # Generate design files for each run
        design_list = []
        for preproc_path in self._sess_preproc:

            # Set run attr, determine template type
            self._get_run(os.path.basename(preproc_path))
            use_short = (
                True
                if self._run == "run-04" or self._run == "run-08"
                else False
            )

            # Make required condition, confound files
            self._make_cond()
            self._make_conf()
            if not self._cond_comm and not self._conf_path:
                continue

            # Make task or lss design files
            if self._model_name == "lss":
                fsl.model.simul_cond_motion(
                    self._subj,
                    self._sess,
                    self._run,
                    self._task,
                    self._subj_work,
                    self._subj_fsl,
                )
                mult_design = make_fsf.write_task_fsf(
                    self._run,
                    preproc_path,
                    self._conf_path,
                    self._cond_comm,
                    use_short,
                    tog_cond=self._tog_cond,
                    lss_cond=self._lss_cond,
                )

                # Account for myriad lss designs
                for design_block in mult_design:
                    for design_path in design_block:
                        design_list.append(design_path)
            else:
                design_path = make_fsf.write_task_fsf(
                    self._run,
                    preproc_path,
                    self._conf_path,
                    self._cond_comm,
                    use_short,
                )

                # Account for failed gen
                if not design_path:
                    continue
                design_list.append(design_path)

        # Execute design files
        self._run_feat(design_list)
        self._push_data()
        self._clean_dcc()

    def _make_cond(self):
        """Generate condition files from BIDS events files for single run."""
        # Find events files in rawdata
        sess_events = sorted(
            glob.glob(f"{self._subj_raw}/*{self._run}*events.tsv")
        )
        if not sess_events or len(sess_events) != 1:
            self._cond_comm = None
            return

        # Generate common condition files
        self._make_cf.load_events(sess_events[0])
        self._cond_comm = self._make_cf.common_events()

        # Generate model-name specific cond files
        if self._model_name == "sep":
            _ = self._make_cf.session_separate_events()
        elif self._model_name == "tog":
            _ = self._make_cf.session_together_events()
        elif self._model_name == "lss":
            self._tog_cond, self._lss_cond = self._make_cf.session_lss_events()

    def _make_conf(self):
        """Generate confounds files from fMRIPrep output for single run."""
        sess_confounds = sorted(
            glob.glob(
                f"{self._subj_fp}/*{self._task}*{self._run}*timeseries.tsv"
            )
        )
        if not sess_confounds or len(sess_confounds) != 1:
            self._conf_path = None
            return

        # Generate confound files
        _, self._conf_path = fsl.model.confounds(
            sess_confounds[0], self._subj_work
        )

    def _run_feat(self, design_list: list):
        """Run FSL FEAT and clean output."""
        _ = Pool().starmap(
            fsl.model.run_feat,
            [
                (
                    fsf_path,
                    self._subj,
                    self._sess,
                    self._model_name,
                    self._log_dir,
                )
                for fsf_path in design_list
            ],
        )
        fsl.helper.clean_up(
            self._subj_work, self._subj_final, self._model_name
        )


# %%
class FslSecond(_SupportFslSecond):
    """Conduct second-level models of EPI data.

    Inherits _SupportFslSecond.

    Coordinate the generation and then feat execution of
    task-based second-level models.

    This workflow will sync with Keoki (via labarserv2), pulling
    required files and uploading workflow output.

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    model_name : str
        Name of FSL model, for keeping condition files and
        output organized
    proj_deriv : path
        Location of project BIDs derivatives, for finding
        preprocessed output
    work_deriv : path
        Output location for intermediates
    log_dir : path
        Output location for log files and scripts
    user_name : str
        User name for DCC, labarserv2
    rsa_key : str, os.PathLike
        Location of RSA key for labarserv2
    keoki_path : str, os.PathLike, optional
        Location of project directory on Keoki

    Methods
    -------
    model_task()
        Generate and execute second-level model for task EPI

    Example
    -------
    wf_fsl = wf_fsl.FslSecond(*args)
    wf_fsl.model_task()

    """

    def __init__(
        self,
        subj,
        sess,
        model_name,
        proj_deriv,
        work_deriv,
        log_dir,
        user_name,
        rsa_key,
        keoki_path="/mnt/keoki/experiments2/EmoRep/Exp2_Compute_Emotion/data_scanner_BIDS",  # noqa: E501
    ):
        """Initialize."""
        if not fsl.helper.valid_name(model_name):
            raise ValueError(f"Unexpected model name : {model_name}")

        print("Initializing FslSecond")
        super().__init__(keoki_path)
        self._subj = subj
        self._sess = sess
        self._model_name = model_name
        self._proj_deriv = proj_deriv
        self._work_deriv = work_deriv
        self._log_dir = log_dir
        self._user_name = user_name
        self._keoki_proj = (
            f"{self._user_name}@{self._keoki_ip}:{self._keoki_path}"
        )
        self._rsa_key = rsa_key
        self._model_level = "second"

    def model_task(self):
        """Run an FSL second-level model for task EPI data.

        Generate and then execute a second-level FSL model. Also
        coordinate data down/upload and cleaning.

        """
        # Make second-level design
        self._setup()
        make_sec = fsl.model.MakeSecondFsf(
            self._subj_work, self._subj_deriv, self._model_name
        )
        design_path = make_sec.write_task_fsf()

        # Execute design file
        _ = fsl.model.run_feat(
            design_path,
            self._subj,
            self._sess,
            self._model_name,
            self._log_dir,
            model_level=self._model_level,
        )

        # Clean up
        fsl.helper.clean_up(
            self._subj_work,
            self._subj_final,
            self._model_name,
        )
        self._push_data()
        self._clean_dcc()


# %%
class FslThirdSubj:
    """Generate reference files for collapsing across subjects.

    Generate text files holding info for the data input, stats evs, and
    stats contrasts fields of the FSL FEAT FMRI analysis GUI. These files
    are used to populate test_build/fsl_group_{task}_template.fsf to
    conduct a third-level FSL analysis which collapses across subject
    to generate group-level task-based emotion maps.

    Output location:
        <proj_dir>/nalyses/model_fsl_group/level-third_name-<model_name>

    Output files:
        level-third_name-*_comb-subj_task-*_data-input.txt
        level-third_name-*_comb-subj_task-*_stats-evs.txt
        level-third_name-*_comb-subj_task-*_stats-contrasts.txt

    Parameters
    ----------
    proj_dir : str, os.PathLike
        Location of project parent directory
    model_name : str, optional
        ["sep"]
        FSL model name
    task_name : str, optional
        ["movies", "scenarios"]
        Value of BIDS task field

    Attributes
    ----------
    task : str
        ["movies", "scenarios"]
        Value of task BIDS field
    input_data : list
        Paths to cope1.feat files for task, input values for
        FSL > FEAT-FMRI > Data > Select FEAT directories
    ev1 : list
        Input values for FSL > FEAT-FMRI > Stats > Full model setup >
        EVs > EV1
    ev1_contrast : list
        Input values for FSL > FEAT-FMRI > Stats > Full model setup >
        Contrast & F-tests > EV1

    Methods
    -------
    build_input_data()
        Generate files with input paths/matrices, builds input_data,
        ev1, ev1_contrast attrs

    Example
    -------
    fg = wf_fsl.FslThirdSubj(*args, **kwargs)
    fg.build_input_data()
    fg.task = "scenarios"
    fg.build_input_data()

    """

    def __init__(self, proj_dir, model_name="sep", task="movies"):
        """Initialize."""
        print("Initializing wf_fsl.FslThirdSubj ...")
        if model_name not in ["sep"]:
            raise ValueError(f"Unsupported model name : {model_name}")

        # Setup
        self._group_dir = os.path.join(
            proj_dir,
            "analyses/model_fsl_group",
            f"level-third_name-{model_name}",
        )
        if not os.path.exists(self._group_dir):
            os.makedirs(self._group_dir)
        self._deriv_dir = os.path.join(
            proj_dir, "data_scanner_BIDS/derivatives/model_fsl"
        )
        self._model_name = model_name

        # Find subjects in derivatives, get paths to all second-level output
        self._subj_list = [
            os.path.basename(x) for x in glob.glob(f"{self._deriv_dir}/sub-*")
        ]
        self._feat_list = glob.glob(
            f"{self._deriv_dir}/sub-*/ses-*/func/level-second_"
            + f"name-{model_name}.gfeat/cope1.feat"
        )
        self.task = task

    def build_input_data(self):
        """Build input_data, ev1, ev1_contrast attrs for task."""
        if self.task not in ["scenarios", "movies"]:
            raise ValueError(f"Unsupported task name : {self.task}")

        self.input_data = []
        for subj in self._subj_list:

            # Find task path
            deriv_path = f"{self._deriv_dir}/{subj}/ses-*/func"
            task_list = glob.glob(
                f"{deriv_path}/condition_files/*{self.task}_*"
            )
            if not task_list:
                continue

            # Add subj/sess if it contains second-level gfeat output
            subj, sess, _task, _run, _desc, _suff = os.path.basename(
                task_list[0]
            ).split("_")
            subj_sess_feat = [
                x for x in self._feat_list if f"{subj}/{sess}" in x
            ]
            if subj_sess_feat:
                self.input_data.append(subj_sess_feat[0])
        self._build_evs()

    def _build_evs(self):
        """Build ev1, ev1_contrast attrs."""
        self.ev1 = [1]
        for _ in self.input_data[1:]:
            self.ev1.append(1.0)
        self.ev1_contrast = [1.0, -1.0]
        self._write_out()

    def _write_out(self):
        """Write input_data, ev1, ev1_contrast to disk."""
        if not hasattr(self, "input_data"):
            self.build_input_data()

        out_data = os.path.join(
            self._group_dir,
            f"level-third_name-{self._model_name}_comb-subj_"
            + f"task-{self.task}_data-input.txt",
        )
        self._write_txt(self.input_data, out_data)

        out_evs = os.path.join(
            self._group_dir,
            f"level-third_name-{self._model_name}_comb-subj_"
            + f"task-{self.task}_stats-evs.txt",
        )
        self._write_txt(self.ev1, out_evs)

        out_con = os.path.join(
            self._group_dir,
            f"level-third_name-{self._model_name}_comb-subj_"
            + f"task-{self.task}_stats-contrasts.txt",
        )
        self._write_txt(self.ev1_contrast, out_con)

    def _write_txt(self, data_list: list, file_path: Union[str, os.PathLike]):
        """Write list to disk."""
        print(f"Writing : {file_path}")
        with open(file_path, "w") as tf:
            for line in data_list:
                tf.write(f"{line}\n")


# %%
class FslThirdSess:
    """Generate reference files for collapsing across sessions.

    Generate files holding info for the data input, stats evs, and
    stats contrasts fields of the FSL FEAT FMRI analysis GUI. These files
    are used to populate test_build/fsl_combine_sessions_template.fsf to
    conduct a third-level FSL analysis which collapses across session
    to generate group-level emotion maps.

    Output location:
        <proj_dir>/analyses/model_fsl_group/level-third_name-<model_name>

    Output files:
        level-third_name-*_comb-sess_data-input.txt
        level-third_name-*_comb-sess_stats-evs.tsv
        level-third_name-*_comb-sess_stats-contrasts.tsv

    Parameters
    ----------
    proj_dir : str, os.PathLike
        Location of project parent directory
    model_name : str, optional
        ["sep"]
        FSL model name

    Attributes
    ----------
    input_data : list
        Paths to cope1.feat files for task, input values for
        FSL > FEAT-FMRI > Data > Select FEAT directories
    df_evs : pd.DataFrame
        Input values for FSL > FEAT-FMRI > Stats > Full model setup > EVs
    df_con : pd.DataFrame
        Input values for FSL > FEAT-FMRI > Stats > Full model setup >
        Contrast & F-tests

    Methods
    -------
    build_input_data()
        Generate files with input paths/matrices, builds input_data,
        df_evs, df_con attrs

    Example
    -------
    fg = wf_fsl.FslThirdSess(*args, **kwargs)
    fg.build_input_data()

    """

    def __init__(self, proj_dir, model_name="sep"):
        """Title."""
        self._proj_dir = proj_dir
        self._model_name = model_name
        self._deriv_dir = os.path.join(
            proj_dir, "data_scanner_BIDS/derivatives/model_fsl"
        )
        self._group_dir = os.path.join(
            proj_dir,
            "analyses/model_fsl_group",
            f"level-third_name-{model_name}",
        )
        if not os.path.exists(self._group_dir):
            os.makedirs(self._group_dir)

    def build_input_data(self):
        """Title."""
        self.input_data = sorted(
            glob.glob(
                f"{self._deriv_dir}/sub-*/ses-*/func/level-second_"
                + f"name-{self._model_name}.gfeat/cope1.feat"
            )
        )
        self._write_txt(
            self.input_data,
            os.path.join(
                self._group_dir,
                f"level-third_name-{self._model_name}_"
                + "comb-sess_data-input.txt",
            ),
        )

        #
        subj_list = [
            x.split("sub-ER")[1].split("/")[0] for x in self.input_data
        ]
        self._unq_subj = sorted(list(set(subj_list)))
        self._build_evs()
        self._build_con()

    def _build_evs(self):
        """Title."""
        self.df_evs = pd.DataFrame(columns=self._unq_subj)
        for cnt, file_path in enumerate(self.input_data):
            subid = file_path.split("sub-ER")[1].split("/")[0]
            self.df_evs.loc[cnt, subid] = 1
        self.df_evs = self.df_evs.fillna(0)
        self._write_tsv(
            self.df_evs,
            os.path.join(
                self._group_dir,
                f"level-third_name-{self._model_name}_comb-sess_stats-evs.tsv",
            ),
        )

    def _build_con(self):
        """Title."""
        self.df_con = pd.DataFrame(columns=self._unq_subj)
        for cnt, subj in enumerate(self._unq_subj):
            self.df_con.loc[cnt, subj] = 1
        self.df_con = self.df_con.fillna(0)
        self._write_tsv(
            self.df_con,
            os.path.join(
                self._group_dir,
                f"level-third_name-{self._model_name}_comb-sess_"
                + "stats-contrasts.tsv",
            ),
        )

    def _write_txt(self, data_list: list, file_path: Union[str, os.PathLike]):
        """Write list to disk."""
        print(f"Writing : {file_path}")
        with open(file_path, "w") as tf:
            for line in data_list:
                tf.write(f"{line}\n")

    def _write_tsv(self, df: pd.DataFrame, file_path: Union[str, os.PathLike]):
        """Write df to disk."""
        print(f"Writing : {file_path}")
        df.to_csv(file_path, sep="\t", index=False)


# %%
def fsl_extract(
    proj_dir,
    subj_list,
    model_name,
    model_level,
    con_name,
    overwrite,
    group_mask="template",
    comb_all=True,
):
    """Extract cope voxel betas and generate dataframe.

    Match cope.nii files to contrast name by mining design.con files
    and then extract each voxel's beta-coefficient and convert them
    into a dataframe.

    A binary brain mask can be generated and used to reduce the
    size of the dataframes.

    Output of group_mask and comb_all are written to:
        <proj_dir>/analyses/model_fsl_group

    Parameters
    ----------
    proj_dir : path
        Location of project directory
    subj_list : list
        Subject IDs to include in dataframe
    model_name : str
        [sep | tog]
        FSL model identifier
    model_level : str
        [first]
        FSL model level
    con_name : str
        [stim | replay | tog]
        Desired contrast from which coefficients will be extracted
            - stim|replay require model_name=sep
            - tog requires model_name=tog
    overwrite : bool
        Whether to overwrite existing beta TSV files
    group_mask : str, optional
        [template]
        Generate a group-level mask, used to identify and remove
        voxels of no interest from beta dataframe
    comb_all : bool, optional
        Combine all participant beta dataframes into an
        omnibus one

    Raises
    ------
    ValueError
        Unexpected values for model_name, model_level, or group_mask

    """
    # Validate parameters
    if model_name not in ["sep", "tog"]:
        raise ValueError(f"Unsupported value for model_name : {model_name}")
    if not fsl.helper.valid_level(model_level):
        raise ValueError(f"Unsupported value for model_level : {model_level}")
    if group_mask not in ["template"]:
        raise ValueError("Unexpected group_mask parameter")
    if not isinstance(overwrite, bool):
        raise TypeError("Expected type bool for overwrite")
    if model_name == "tog" and con_name != "tog":
        raise ValueError(
            f"Unsupported con_name for model_name={model_name} : {con_name}"
        )
    if model_name == "sep" and (con_name != "stim" and con_name != "replay"):
        raise ValueError(
            f"Unsupported con_name for model_name={model_name} : {con_name}"
        )

    # Orient to project directory
    out_dir = os.path.join(proj_dir, "analyses/model_fsl_group")
    proj_deriv = os.path.join(proj_dir, "data_scanner_BIDS", "derivatives")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize beta extraction, generate mask and identify
    # censor coordinates
    get_betas = fsl.group.ExtractTaskBetas(proj_dir)
    mask_path = afni.masks.tpl_gm(out_dir)
    get_betas.mask_coord(mask_path)

    def _get_betas(subj: str, sess: str, subj_dir: Union[str, os.PathLike]):
        """Flatten MRI beta matrix."""
        # Identify task name from condition files
        subj_func_dir = os.path.join(subj_dir, sess, "func")
        task_path = glob.glob(f"{subj_func_dir}/condition_files/*_events.txt")[
            0
        ]
        _subj, _sess, task, _run, _desc, _suff = os.path.basename(
            task_path
        ).split("_")

        # Identify all design.con files generated by FEAT, generate
        # session dataframe
        design_list = sorted(
            glob.glob(
                f"{subj_func_dir}/run-*{model_level}*{model_name}*"
                + "/design.con"
            )
        )
        if not design_list:
            return
        _ = get_betas.make_func_matrix(
            subj,
            sess,
            task,
            model_name,
            model_level,
            con_name,
            design_list,
            subj_func_dir,
            overwrite=overwrite,
        )

    # Make beta dataframe for each subject, session
    for subj in subj_list:
        subj_dir = os.path.join(proj_deriv, "model_fsl", subj)
        sess_list = [
            os.path.basename(x) for x in sorted(glob.glob(f"{subj_dir}/ses-*"))
        ]

        # Run sessions in parallel
        mult_proc = [
            Process(
                target=_get_betas,
                args=(
                    subj,
                    sess,
                    subj_dir,
                ),
            )
            for sess in sess_list
        ]
        for proc in mult_proc:
            proc.start()
        for proc in mult_proc:
            proc.join()
        print("\t\tDone", flush=True)

    # Combine all participant dataframes
    if comb_all:
        _, _ = fsl.group.comb_matrices(
            subj_list, model_name, model_level, con_name, proj_deriv, out_dir
        )


# %%
def fsl_classify_mask(
    proj_dir, model_name, model_level, con_name, task_name, tpl_path
):
    """Convert a dataframe of classifier values into NIfTI masks.

    Orient to NIfTI header and matrix size from a template, then
    iterate through each row of the classifier importance output
    and make a mask in template space from the row values.

    Check for data in:
        <proj-dir>/analyses/classify_fMRI_plsda/classifier_output

    and writes output to:
        <proj-dir>/analyses/classify_fMRI_plsda/voxel_importance_maps

    Parameters
    ----------
    proj_dir : path
        Location of project directory
    model_name : str
        [sep | tog]
        FSL model identifier
    model_level : str
        [first]
        FSL model level
    con_name : str
        [stim | replay | tog]
        Contrast name of extracted coefficients
    task_name : str
        [movies | scenarios | all]
        Name of stimulus type
    tpl_path : path
        Location and name of template

    Raises
    ------
    FileNotFoundError
        Missing template
        Missing dataframe, or wrong dataframe name
    ValueError
        Unexpected user-specified parameters
        Unexpected number of emotions/rows in dataframe

    """
    # Check user input
    if model_name not in ["sep", "tog"]:
        raise ValueError(f"Unsupported model name : {model_name}")
    if model_level != "first":
        raise ValueError(f"Unsupported model level : {model_level}")
    if not fsl.helper.valid_contrast(con_name):
        raise ValueError(f"Unsupported contrast name : {con_name}")
    if task_name not in ["movies", "scenarios", "all"]:
        raise ValueError(f"Unexpected value for task : {task_name}")
    if not os.path.exists(tpl_path):
        raise FileNotFoundError(f"Missing file : {tpl_path}")

    # Identify classifier dataframe
    data_dir = os.path.join(
        proj_dir, "analyses/classify_fMRI_plsda/classifier_output"
    )
    class_path = os.path.join(
        data_dir,
        f"level-{model_level}_name-{model_name}_task-{task_name}_"
        + f"con-{con_name}Washout_voxel-importance.tsv",
    )
    if not os.path.exists(class_path):
        raise FileNotFoundError(
            f"Check filename -- failed to find : {class_path}"
        )

    # Load, check dataframe
    df_import = pd.read_csv(class_path, sep="\t")
    emo_list = df_import["emo_id"].tolist()
    if len(emo_list) != 15:
        raise ValueError(
            f"Unexpected number of emotions from df.emo_id : {emo_list}"
        )

    # Make a mask for each emotion in dataframe
    out_dir = os.path.join(
        proj_dir, "analyses/classify_fMRI_plsda/voxel_importance_maps"
    )
    mk_mask = fsl.group.ImportanceMask()
    mk_mask.mine_template(tpl_path)
    mask_list = []
    for emo_name in emo_list:
        print(f"Making importance mask for : {emo_name}")
        df_emo = df_import[df_import["emo_id"] == emo_name]
        df_emo = df_emo.drop("emo_id", axis=1).reset_index(drop=True)
        mask_path = os.path.join(
            out_dir,
            f"level-{model_level}_name-{model_name}_task-{task_name}_"
            + f"con-{con_name}Washout_emo-{emo_name}_map.nii.gz",
        )
        _ = mk_mask.make_mask(df_emo, mask_path)
        mask_list.append(mask_path)

    # Trigger conjunction analyses
    conj = fsl.group.ConjunctAnalysis(mask_list, out_dir)
    conj.omni_map()
    conj.arousal_map()
    conj.valence_map()
