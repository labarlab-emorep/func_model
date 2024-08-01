"""FSL-based workflows.

FslFirst : GLM task, rest data via FSL
FslSecond : second-level GLM of task data via FSl
FslThirdSubj : generate needed info for third-level analyses which
                collapses across subject
FslThirdSess : generate needed info for third-level analyses which
                collapse across session
FslFourthSess : generate needed info for fourth-level analyses which
                finishes the collapse across sessions started by
                FslThirdSess
ExtractBetas : extract task beta-coefficiens from FSL GLM
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
from natsort import natsorted
from func_model.resources import masks
from func_model.resources import model
from func_model.resources import group
from func_model.resources import helper


# %%
class _SupportFsl:
    """General helper methods for first- and second-level workflows."""

    def __init__(self, keoki_path: Union[str, os.PathLike]):
        """Initialize."""
        try:
            self._rsa_key = os.environ["RSA_LS2"]
        except KeyError as e:
            raise Exception(
                "Missing required environmental variable RSA_LS2"
            ) from e
        self._keoki_path = keoki_path

    @property
    def _ls2_ip(self):
        """Return labarserv2 ip addr."""
        return "ccn-labarserv2.vm.duke.edu"

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
        h_sp = subprocess.Popen(
            bash_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
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
                {os.environ["USER"]}@{self._ls2_ip} \
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
        """Set attr sess_preproc for paths to preproc EPIs."""
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
        if not helper.valid_task(self._task):
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
class _SupportFslThirdFourth:
    """Methods for third- and fourth-level FSL classes.

    Methods
    -------
    write_tsv()
        Write pd.DataFrame to TSV
    write_txt()
        Write list to TXT

    """

    def write_txt(self, data_list: list, file_path: Union[str, os.PathLike]):
        """Write list to disk."""
        print(f"Writing : {file_path}")
        with open(file_path, "w") as tf:
            for line in data_list:
                tf.write(f"{line}\n")

    def write_tsv(self, df: pd.DataFrame, file_path: Union[str, os.PathLike]):
        """Write df to disk."""
        print(f"Writing : {file_path}")
        df.to_csv(file_path, sep="\t", index=False)


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
        keoki_path="/mnt/keoki/experiments2/EmoRep/Exp2_Compute_Emotion/data_scanner_BIDS",  # noqa: E501
    ):
        """Initialize."""
        if not helper.valid_name(model_name):
            raise ValueError(f"Unexpected model name : {model_name}")
        if not helper.valid_preproc(preproc_type):
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
        self._keoki_proj = (
            f"{os.environ['USER']}@{self._ls2_ip}:{self._keoki_path}"
        )

    def model_rest(self):
        """Run an FSL first-level model for resting EPI data.

        Generate required a confounds and design file, then
        use FSL's FEAT to run a first-level model.

        """
        # Setup and get required data
        print("\tRunning first-level rest model")
        self._setup()

        # Initialize needed classes, find preprocessed resting EPI
        make_fsf = model.MakeFirstFsf(
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
        make_fsf = model.MakeFirstFsf(
            self._subj_work, self._proj_deriv, self._model_name
        )
        self._make_cf = model.ConditionFiles(
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
                model.simul_cond_motion(
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
        _, self._conf_path = model.confounds(
            sess_confounds[0], self._subj_work
        )

    def _run_feat(self, design_list: list):
        """Multiprocess FSL FEAT and clean output."""
        c_size = 10 if self._model_name == "lss" else None
        with Pool() as pool:
            items = [
                (
                    fsf_path,
                    self._subj,
                    self._sess,
                    self._model_name,
                    self._log_dir,
                )
                for fsf_path in design_list
            ]
            _ = pool.starmap(model.run_feat, items, chunksize=c_size)
        helper.clean_up(self._subj_work, self._subj_final, self._model_name)


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
        keoki_path="/mnt/keoki/experiments2/EmoRep/Exp2_Compute_Emotion/data_scanner_BIDS",  # noqa: E501
    ):
        """Initialize."""
        if not helper.valid_name(model_name):
            raise ValueError(f"Unexpected model name : {model_name}")

        print("Initializing FslSecond")
        super().__init__(keoki_path)
        self._subj = subj
        self._sess = sess
        self._model_name = model_name
        self._proj_deriv = proj_deriv
        self._work_deriv = work_deriv
        self._log_dir = log_dir
        self._keoki_proj = (
            f"{os.environ['USER']}@{self._ls2_ip}:{self._keoki_path}"
        )
        self._model_level = "second"

    def model_task(self):
        """Run an FSL second-level model for task EPI data.

        Generate and then execute a second-level FSL model. Also
        coordinate data down/upload and cleaning.

        """
        # Make second-level design
        self._setup()
        make_sec = model.MakeSecondFsf(
            self._subj_work, self._subj_deriv, self._model_name
        )
        design_path = make_sec.write_task_fsf()

        # Execute design file
        _ = model.run_feat(
            design_path,
            self._subj,
            self._sess,
            self._model_name,
            self._log_dir,
            model_level=self._model_level,
        )

        # Clean up
        helper.clean_up(
            self._subj_work,
            self._subj_final,
            self._model_name,
        )
        self._push_data()
        self._clean_dcc()


# %%
class FslThirdSubj(_SupportFslThirdFourth):
    """Generate reference files for collapsing across subjects.

    Inherits _SupportFslThirdFourth.

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
    task_name : str
        ["movies", "scenarios"]
        Value of BIDS task field
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

        # Find data for each subject
        print("Building input_data, ev1, and ev1_contrast ...")
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
        self.write_txt(
            self.input_data,
            os.path.join(
                self._group_dir,
                f"level-third_name-{self._model_name}_comb-subj_"
                + f"task-{self.task}_data-input.txt",
            ),
        )
        self.write_txt(
            self.ev1,
            os.path.join(
                self._group_dir,
                f"level-third_name-{self._model_name}_comb-subj_"
                + f"task-{self.task}_stats-evs.txt",
            ),
        )
        self.write_txt(
            self.ev1_contrast,
            os.path.join(
                self._group_dir,
                f"level-third_name-{self._model_name}_comb-subj_"
                + f"task-{self.task}_stats-contrasts.txt",
            ),
        )


# %%
class FslThirdSess(_SupportFslThirdFourth):
    """Generate reference files for collapsing across sessions.

    Inherits _SupportFslThirdFourth.

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
        """Initialize."""
        print("Initializing wf_fsl.FslThirdSess ...")

        # Validate
        if model_name not in ["sep"]:
            raise ValueError(f"Unsupported model name : {model_name}")

        # Setup
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
        """Build input_data, df_evs, df_con attrs."""
        print("Building input_data, df_con, df_evs ...")

        # Find all input data
        self.input_data = sorted(
            glob.glob(
                f"{self._deriv_dir}/sub-*/ses-*/func/level-second_"
                + f"name-{self._model_name}.gfeat/cope1.feat"
            )
        )
        self.write_txt(
            self.input_data,
            os.path.join(
                self._group_dir,
                f"level-third_name-{self._model_name}_"
                + "comb-sess_data-input.txt",
            ),
        )

        # Find all subjects, trigger evs, con
        subj_list = [
            x.split("sub-ER")[1].split("/")[0] for x in self.input_data
        ]
        self._unq_subj = sorted(list(set(subj_list)))
        self._build_evs()
        self._build_con()

    def _build_evs(self):
        """Build df_evs attr."""
        # Contrast dataframe where column=subj and rows=sess
        self.df_evs = pd.DataFrame(columns=self._unq_subj)
        for cnt, file_path in enumerate(self.input_data):
            subid = file_path.split("sub-ER")[1].split("/")[0]
            self.df_evs.loc[cnt, subid] = 1
        self.df_evs = self.df_evs.fillna(0)
        self.write_tsv(
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
        self.write_tsv(
            self.df_con,
            os.path.join(
                self._group_dir,
                f"level-third_name-{self._model_name}_comb-sess_"
                + "stats-contrasts.tsv",
            ),
        )


# %%
class FslFourthSess(_SupportFslThirdFourth):
    """Generate reference files for collapsing across sessions.

    Inherits _SupportFslThirdFourth.

    TODO

    Output location:
        <proj_dir>/analyses/model_fsl_group/level-fourth_name-<model_name>

    Output files:
        level-fourth_name-*_comb-sess_data-input.txt
        level-fourth_name-*_comb-sess_stats-evs.tsv
        level-fourth_name-*_comb-sess_stats-contrasts.tsv

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
    df_evs : pd.DataFrame
    con


    Methods
    -------
    build_input_data()


    Example
    -------


    """

    def __init__(self, proj_dir, model_name="sep"):
        """Title."""
        self._proj_dir = proj_dir
        self._model_name = model_name
        self._deriv_dir = os.path.join(
            proj_dir, "data_scanner_BIDS/derivatives/model_fsl"
        )
        group_dir = os.path.join(proj_dir, "analyses/model_fsl_group")
        self._data_dir = os.path.join(
            group_dir,
            f"level-third_name-{model_name}",
            f"level-third_name-{model_name}_comb-sess.gfeat",
        )
        if not os.path.exists(self._data_dir):
            raise FileNotFoundError(f"Expected FEAT output : {self._data_dir}")
        self._group_dir = os.path.join(
            group_dir,
            f"level-fourth_name-{model_name}",
        )
        if not os.path.exists(self._group_dir):
            os.makedirs(self._group_dir)

    def build_input_data(self):
        """Title."""
        print(f"Finding data in : {self._data_dir}")
        self.input_data = sorted(
            glob.glob(f"{self._data_dir}/cope*.feat/stats/cope*.nii.gz"),
            key=len,
        )
        self.input_data = natsorted(self.input_data)
        if not self.input_data:
            raise ValueError(f"Expected FEAT output : {self._data_dir}")
        self.write_txt(
            self.input_data,
            os.path.join(
                self._group_dir,
                f"level-fourth_name-{self._model_name}_"
                + "comb-sess_data-input.txt",
            ),
        )
        self._build_evs()
        self._build_con()

    def _build_evs(self):
        """Title."""
        #
        cope_all = [x.split("/")[-3] for x in self.input_data]
        cope_list = [x for x in natsorted(list(set(cope_all)))]
        emo_list = [
            "Amusement",
            "Anger",
            "Anxiety",
            "Awe",
            "Calmness",
            "Craving",
            "Disgust",
            "Excitement",
            "Fear",
            "Horror",
            "Joy",
            "Neutral",
            "Romance",
            "Sadness",
            "Surprise",
        ]
        if len(cope_list) != len(emo_list):
            raise ValueError("Unexpected number of emotions")
        self._cope_dict = {x: y for x, y in zip(emo_list, cope_list)}

        #
        subj_all = [os.path.basename(x) for x in self.input_data]
        subj_list = [x for x in natsorted(list(set(subj_all)))]

        #
        self.df_evs = pd.DataFrame(columns=emo_list)
        cnt_row = 0
        for col in emo_list:
            for _ in subj_list:
                self.df_evs.loc[cnt_row, col] = 1
                cnt_row += 1
        self.df_evs = self.df_evs.fillna(0)
        self.write_tsv(
            self.df_evs,
            os.path.join(
                self._group_dir,
                f"level-fourth_name-{self._model_name}_"
                + "comb-sess_stats-evs.tsv",
            ),
        )

    def _build_con(self):
        """Title."""
        self.df_con = pd.DataFrame(columns=self._cope_dict.keys())
        for row, col in enumerate(self._cope_dict.keys()):
            self.df_con.loc[row, col] = 1
        self.df_con = self.df_con.fillna(0)
        self.write_tsv(
            self.df_con,
            os.path.join(
                self._group_dir,
                f"level-fourth_name-{self._model_name}_comb-sess_"
                + "stats-contrasts.tsv",
            ),
        )


class ExtractBetas:
    """Extract cope voxel betas from FSL model.

    Match cope.nii files to contrast name by mining design.con files
    and then extract each voxel's beta-coefficient. Data are then
    sent to MySQL db_emorep.

    A binary brain mask is also generated and used to reduce the
    size of the dataframes (900K -> 137K voxels).

    Parameters
    ----------
    proj_dir : path
        Location of project directory
    subj_list : list
        Subject IDs to include in dataframe
    model_name : str
        {"lss", "sep"}
        FSL model identifier
    con_name : str
        {"stim", "tog"}
        Desired contrast from which coefficients will be extracted
    overwrite : bool
        Whether to overwrite existing data

    Example
    -------
    ex_reg = ExtractBetas(*args)
    ex_reg.get_betas()

    """

    def __init__(
        self,
        proj_dir,
        subj_list,
        model_name,
        con_name,
        overwrite,
    ):
        """Initialize."""
        self._proj_dir = proj_dir
        self._subj_list = subj_list
        self._model_name = model_name
        self._con_name = con_name
        self._overwrite = overwrite

        # Set attrs for future flexibility, get extraction methods
        self._model_level = "first"
        self._group_mask = "template"
        self._ex_betas = group.ExtractTaskBetas()

    def _validate(self):
        """Validate user-specified arguments for beta extraction."""
        # Validate user arguments
        if self._model_name not in ["sep", "lss"]:
            raise ValueError(
                f"Unsupported value for model_name : {self._model_name}"
            )
        if self._model_level not in ["first"]:
            raise ValueError(
                f"Unsupported value for model_level : {self._model_level}"
            )
        if self._group_mask not in ["template"]:
            raise ValueError("Unexpected group_mask parameter")

        # Validate argument combinations
        if self._model_name == "sep" and self._con_name != "stim":
            raise ValueError(
                "Unexpected model contrast pair : "
                + f"{self._model_name}, {self._con_name}"
            )
        if self._model_name == "lss" and self._con_name != "tog":
            raise ValueError(
                "Unexpected model contrast pair : "
                + f"{self._model_name}, {self._con_name}"
            )

    def _setup(self):
        """Set attrs out_dir, proj_deriv and make dirs."""
        self._out_dir = os.path.join(
            self._proj_dir, "analyses/model_fsl_group"
        )
        self._proj_deriv = os.path.join(
            self._proj_dir, "data_scanner_BIDS/derivatives"
        )
        if not os.path.exists(self._out_dir):
            os.makedirs(self._out_dir)

    def get_betas(self):
        """Extract betas from all detected FSL output.

        Determine mask voxels (used for reducing beta matrix size),
        then extract betas for all FSL model output found in each
        session. Sessions are multiprocessed for non-LSS models.

        """
        # Validate and setup
        self._validate()
        self._setup()

        # Get mask coordinates (set attr rm_voxel)
        mask_path = masks.tpl_gm(self._out_dir)
        self._ex_betas.mask_coord(mask_path)

        # Identify sessions
        model_dir = (
            "model_fsl-lss" if self._model_name == "lss" else "model_fsl"
        )
        for self._subj in self._subj_list:
            self._subj_dir = os.path.join(
                self._proj_deriv, model_dir, self._subj
            )
            sess_list = [
                os.path.basename(x)
                for x in sorted(glob.glob(f"{self._subj_dir}/ses-*"))
            ]

            # Multiproc sessions for non-LSS models, run LSS sessions
            # serially due to large number of beta extractions.
            if self._model_name == "lss":
                for sess in sess_list:
                    self._mine_betas(sess)
            else:
                self._mult_proc(sess_list)

    def _mult_proc(self, sess_list: list):
        """Run session mining in parallel."""
        mult_proc = [
            Process(
                target=self._mine_betas,
                args=(sess,),
            )
            for sess in sess_list
        ]
        for proc in mult_proc:
            proc.start()
            proc.join()
        print("\tDone", flush=True)

    def _mine_betas(self, sess: str):
        """Flatten MRI beta matrix."""
        # Identify task name from condition files
        subj_func = os.path.join(self._subj_dir, sess, "func")
        cond_all = glob.glob(f"{subj_func}/condition_files/*_event*.txt")
        if not cond_all:
            return
        _subj, _sess, task, _run, _desc, _suff = os.path.basename(
            cond_all[0]
        ).split("_")

        # Identify all design.con files generated by FEAT, generate
        # session dataframe.
        design_list = sorted(
            glob.glob(
                f"{subj_func}/run-*{self._model_level}*{self._model_name}*"
                + "/design.con"
            )
        )
        if not design_list:
            return
        self._ex_betas.make_beta_matrix(
            self._subj,
            sess,
            task,
            self._model_name,
            self._model_level,
            self._con_name,
            design_list,
            subj_func,
            self._overwrite,
        )


# %%
def fsl_classify_mask(
    proj_dir, model_name, model_level, con_name, task_name, tpl_path
):
    """Convert a dataframe of classifier values into NIfTI masks.

    Make binary cluster maps in MNI space from emo_* fields of
    db_emorep.tbl_plsda_binary_gm. Derives NIfTI header and matrix
    from tpl_path.

    Parameters
    ----------
    proj_dir : str, os.PathLike
        Location of project directory
    model_name : str
        {"sep", "tog"}
        FSL model identifier
    model_level : str
        {"first"}
        FSL model level
    con_name : str
        {"stim", "tog", "replay"}
        Contrast name of extracted coefficients
    task_name : str
        {"movies", "scenarios", "both"}
        Name of stimulus type
    tpl_path : str, os.PathLike
        Location and name of template

    """
    # Check user input
    if model_name not in ["sep", "tog"]:
        raise ValueError(f"Unsupported model name : {model_name}")
    if model_level != "first":
        raise ValueError(f"Unsupported model level : {model_level}")
    if not helper.valid_contrast(con_name):
        raise ValueError(f"Unsupported contrast name : {con_name}")
    if task_name not in ["movies", "scenarios", "both"]:
        raise ValueError(f"Unexpected value for task : {task_name}")
    if not os.path.exists(tpl_path):
        raise FileNotFoundError(f"Missing file : {tpl_path}")

    # Setup output location
    out_dir = os.path.join(
        proj_dir,
        "analyses/classify_fMRI_plsda/voxel_importance_maps",
        f"name-{model_name}_task-{task_name}_maps",
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Derive MNI header info
    mk_mask = group.ImportanceMask(tpl_path)

    # Generate binary mask and cluster for each
    # db_emorep.tbl_plsda_binary_gm.emo_* field
    mask_list = []
    emo_list = mk_mask.emo_names()
    for emo_name in emo_list:
        print(f"Making importance mask for : {emo_name}")
        mask_list.append(
            mk_mask.sql_masks(
                task_name,
                model_name,
                con_name,
                emo_name,
                "binary",
                out_dir,
                cluster=True,
            )
        )

    # Trigger conjunction analyses
    conj = group.ConjunctAnalysis(mask_list, out_dir)
    conj.omni_map()
    conj.arousal_map()
    conj.valence_map()
