"""Helper methods for AFNI-based pipelines.

prepend_afni_sing : setup singularity call for afni
valid_task : check task name
valid_models : check model name
valid_univ_test : check univariate test type
valid_mvm_test : check multivariate test type
emo_switch : supply emotion names

"""

import os
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
