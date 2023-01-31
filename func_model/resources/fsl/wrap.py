"""Title."""
import os
import glob
import nibabel as nib
from . import model


class MakeCondConf:
    """Title."""

    def __init__(self, subj, sess, subj_work):
        """Title."""
        self.subj = subj
        self.sess = sess
        self.subj_work = subj_work

    def make_condition(self, model_name, proj_rawdata):
        """Title."""
        #
        subj_sess_raw = os.path.join(proj_rawdata, self.subj, self.sess)
        sess_events = sorted(glob.glob(f"{subj_sess_raw}/func/*events.tsv"))
        if not sess_events:
            raise FileNotFoundError(
                f"Expected BIDs events files in {subj_sess_raw}"
            )

        #
        _task_short = (
            os.path.basename(sess_events[0]).split("task-")[-1].split("_")[0]
        )
        task = "task-" + _task_short
        if task not in ["task-movies", "task-scenarios"]:
            raise ValueError(f"Unexpected task name : {task}")

        #
        make_cf = model.ConditionFiles(
            self.subj, self.sess, task, self.subj_work, sess_events
        )
        for run_num in make_cf.run_list:
            make_cf.common_events(run_num)
            if model_name == "sep":
                make_cf.session_separate_events(run_num)
        return task

    def make_confound(self, task, proj_deriv):
        """Title.

        Desc.

        """
        #
        fp_subj_sess = os.path.join(
            proj_deriv, "pre_processing/fmriprep", self.subj, self.sess
        )
        sess_confounds = sorted(
            glob.glob(f"{fp_subj_sess}/func/*{task}*timeseries.tsv")
        )
        if not sess_confounds:
            raise FileNotFoundError(
                f"Expected fMRIPrep confounds files in {fp_subj_sess}"
            )

        #
        for conf_path in sess_confounds:
            _ = model.confounds(conf_path, self.subj_work)


def write_fsf(
    subj, sess, task, model_name, model_level, subj_work, proj_deriv
):
    """Title."""

    def _get_run(file_name: str) -> str:
        "Return run field from temporal filtered filename."
        try:
            _su, _se, _ta, run, _sp, _re, _de, _su = file_name.split("_")
        except IndexError:
            raise ValueError(
                "Improperly formatted file name for preprocessed BOLD."
            )
        return run

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

    #
    make_fsf = model.MakeFsf(
        subj, sess, task, subj_work, proj_deriv, model_name, model_level
    )
    make_fsf.load_template()
    for preproc_path in sess_preproc:

        #
        run = _get_run(os.path.basename(preproc_path))
        confound_path = glob.glob(f"{subj_work}/confounds_files/*{run}*.txt")[
            0
        ]

        #
        img = nib.load(preproc_path)
        img_header = img.header
        num_vol = img_header.get_data_shape()[3]
        del img

        #
        use_short = True if run == "run-04" or run == "run-08" else False
        make_fsf.write_first_fsf(
            run, num_vol, preproc_path, confound_path, use_short
        )
