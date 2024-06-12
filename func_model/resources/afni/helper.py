"""Helper methods for AFNI-based pipelines."""

import os
import glob
import shutil
import subprocess
from typing import Union


def prepend_afni_sing(
    work_deriv: Union[str, os.PathLike], subj_work: Union[str, os.PathLike]
) -> list:
    """Supply singularity call for AFNI."""
    try:
        sing_afni = os.environ["SING_AFNI"]
    except KeyError as e:
        print("Missing required variable SING_AFNI")
        raise e

    return [
        "singularity run",
        "--cleanenv",
        f"--bind {work_deriv}:{work_deriv}",
        f"--bind {subj_work}:{subj_work}",
        f"--bind {subj_work}:/opt/home",
        sing_afni,
    ]


def valid_task(task_name: str) -> bool:
    """Return bool of whether task_name is supported."""
    return task_name in ["task-movies", "task-scenarios"]


def valid_models(model_name: str) -> bool:
    """Return bool of whether model_name is supported."""
    return model_name in ["univ", "rest", "mixed"]


def valid_univ_test(test_name: str) -> bool:
    """Return bool of whether test_name is supported."""
    return test_name in ["student", "paired"]


def valid_mvm_test(test_name: str) -> bool:
    """Return bool of whether test_name is supported."""
    return test_name in ["rm"]


def emo_switch() -> dict:
    """Return events-AFNI emotion mappings."""
    return {
        "amusement": "Amu",
        "anger": "Ang",
        "anxiety": "Anx",
        "awe": "Awe",
        "calmness": "Cal",
        "craving": "Cra",
        "disgust": "Dis",
        "excitement": "Exc",
        "fear": "Fea",
        "horror": "Hor",
        "joy": "Joy",
        "neutral": "Neu",
        "romance": "Rom",
        "sadness": "Sad",
        "surprise": "Sur",
    }


class MoveFinal:
    """Copy final files from /work to /group.

    Identify desired files in /work and copy them
    to /group. Then purge /work.

    Methods
    -------
    copy_files(save_list)
        Copy list of files from /work to /group

    """

    def __init__(
        self, subj, sess, proj_deriv, subj_work, sess_anat, model_name
    ):
        """Copy files from work to group.

        Initiate object, construct list of desired files, then
        copy them to /group. Clean up /work.

        Parameters
        ----------
        subj : str
            BIDs subject identifier
        sess : str
            BIDs session identifier
        proj_deriv : path
            Location of project derivatives
        subj_work : path
            Location of working directory for intermediates
        sess_anat : dict
            Contains reference names (key) and paths (value) to
            preprocessed anatomical files.
            Required keys:
            -   [mask-WMe] = path to eroded CSF mask
            -   [mask-int] = path to intersection mask
        model_name : str
            Desired AFNI model, for triggering different workflows

        Raises
        ------
        KeyError
            Missing required key in sess_anat

        """
        # Validate dict keys
        for _key in ["mask-WMe", "mask-int"]:
            if _key not in sess_anat:
                raise KeyError(f"Expected {_key} key in sess_anat")

        # Set attributes
        self._subj = subj
        self._sess = sess
        self._proj_deriv = proj_deriv
        self._subj_work = subj_work
        self._sess_anat = sess_anat
        self._model_name = model_name

        # Trigger list construction, copy files
        save_list = (
            self._make_list_rest()
            if model_name == "rest"
            else self._make_list_task()
        )
        self.copy_files(save_list)

    def _make_list_task(self):
        """Find AFNI task files for saving.

        Files found:
        -   motion_files directory
        -   timing_files directory
        -   decon_<model_name>_stats_REML+tlrc.*
        -   decon_<model_name>.sh
        -   X.decon_<model_name>.*
        -   WM, intersection masks

        Returns
        -------
        list

        Raises
        ------
        FileNotFoundError
            decon_<model_name>_stats_REML+tlrc.* files were not found

        """
        subj_motion = os.path.join(self._subj_work, "motion_files")
        subj_timing = os.path.join(self._subj_work, "timing_files")
        stat_list = glob.glob(
            f"{self._subj_work}/decon_{self._model_name}_stats_REML+tlrc.*"
        )
        if stat_list:
            sh_list = glob.glob(
                f"{self._subj_work}/decon_{self._model_name}*.sh"
            )
            x_list = glob.glob(
                f"{self._subj_work}/X.decon_{self._model_name}.*"
            )
            save_list = stat_list + sh_list + x_list
            save_list.append(self._sess_anat["mask-WMe"])
            save_list.append(self._sess_anat["mask-int"])
            save_list.append(subj_motion)
            save_list.append(subj_timing)
        else:
            raise FileNotFoundError(
                f"Missing decon_{self._model_name} files in {self._subj_work}"
            )
        return save_list

    def _make_list_rest(self):
        """Find AFNI rest files for saving.

        Files found:
        -   decon_rest_anaticor+tlrc.*
        -   decon_rest.sh
        -   X.decon_rest.*
        -   Seed files
        -   Intersection mask

        Returns
        -------
        list

        Raises
        ------
        FileNotFoundError
            decon_rest_anaticor+tlrc.* files were not found

        """
        stat_list = glob.glob(f"{self._subj_work}/decon_rest_anaticor*+tlrc.*")
        if stat_list:
            seed_list = glob.glob(f"{self._subj_work}/seed_*")
            x_list = glob.glob(f"{self._subj_work}/X.decon_rest.*")
            save_list = stat_list + seed_list + x_list
            save_list.append(self._sess_anat["mask-int"])
            save_list.append(f"{self._subj_work}/decon_rest.sh")
        else:
            raise FileNotFoundError(
                f"Missing decon_rest files in {self._subj_work}"
            )
        return save_list

    def copy_files(self, save_list):
        """Copy desired files from /work to /group.

        Use bash subprocess of copy for speed, delete
        files in /work after copy has happened. Files
        are copied to:
            <proj_deriv>/model_afni/<subj>/<sess>/func

        Parameters
        ----------
        save_list : list
            Paths to files in /work that should be saved

        """
        # Setup save location in group directory
        subj_final = os.path.join(
            self._proj_deriv, "model_afni", self._subj, self._sess, "func"
        )
        if not os.path.exists(subj_final):
            os.makedirs(subj_final)

        # Copy desired files to group location
        for h_save in save_list:
            bash_cmd = f"cp -r {h_save} {subj_final}"
            h_sp = subprocess.Popen(
                bash_cmd, shell=True, stdout=subprocess.PIPE
            )
            _ = h_sp.communicate()
            h_sp.wait()
            chk_save = os.path.join(subj_final, h_save)
            if not os.path.exists(chk_save):
                raise FileNotFoundError(f"Expected to find {chk_save}")

        # Clean up - remove session directory in case
        # other session is still running.
        shutil.rmtree(os.path.dirname(self._subj_work))
