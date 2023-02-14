"""Methods for group-level analyses."""
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from func_model.resources.fsl import helper
from func_model.resources.general import matrix


class ExtractTaskBetas(matrix.NiftiArray):
    """Title.

    Desc.

    Methods
    -------

    """

    def __init__(self, proj_dir, float_prec=4):
        """Initialize.

        Parameters
        ----------
        proj_dir : path
            Location of project directory
        float_prec : int, optional
            Desired float point precision of dataframes

        Raises
        ------
        TypeError
            Unexpected type for float_prec

        """
        print("Initializing ExtractTaskBetas")
        super().__init__(float_prec)
        self._proj_dir = proj_dir

    def _read_contrast(self):
        """Match contrast name to number, set as self._con_dict."""
        # Extract design.con lines starting with /ContrastName
        con_lines = []
        with open(self._design_path) as dp:
            for ln in dp:
                if ln.startswith("/ContrastName"):
                    con_lines.append(ln[1:])
        if len(con_lines) == 0:
            raise ValueError(
                "Could not find ContrastName in " + f"{self._design_path}"
            )

        # Organize contrast lines
        self._con_dict = {}
        for line in con_lines:
            con, name = line.split()
            self._con_dict[name] = con

    def _drop_replay(self):
        """Remove replay* and keep stim* keys from self._con_dict."""
        if not hasattr(self, "_con_dict"):
            raise AttributeError(
                "Missing self._con_dict, try self._read_contrast"
            )
        for key in list(self._con_dict.keys()):
            if key.startswith("replay"):
                del self._con_dict[key]

    def _clean_contrast(self):
        """Clean self._con_dict into pairs of emotion: contrast num."""
        if not hasattr(self, "_con_dict"):
            raise AttributeError(
                "Missing self._con_dict, try self._read_contrast"
            )
        out_dict = {}
        for key, value in self._con_dict.items():
            new_key = key[4:].split("GT")[0].lower()
            out_dict[new_key] = value[-1]
        self._con_dict = out_dict

    def _find_copes(self) -> dict:
        """Match emotion name to cope file."""
        # Mine, organize design.con info
        self._read_contrast()
        self._drop_replay()
        self._clean_contrast()

        # Orient from desing.con to stats dir
        feat_dir = os.path.dirname(self._design_path)
        stats_dir = os.path.join(feat_dir, "stats")

        # Match emotion to nii path
        cope_dict = {}
        for emo, num in self._con_dict.items():
            nii_path = os.path.join(stats_dir, f"cope{num}.nii.gz")
            if not os.path.exists(nii_path):
                raise FileNotFoundError(
                    f"Missing expected cope file : {nii_path}"
                )
            cope_dict[emo] = nii_path
        return cope_dict

    def make_func_matrix(
        self,
        subj,
        sess,
        task,
        model_name,
        model_level,
        design_list,
        subj_deriv_func,
    ):
        """Title."""
        # Check input, set attributes and output location
        if not helper.valid_task(task):
            raise ValueError(f"Unexpected value for task : {task}")

        print(f"\tGetting betas from {subj}, {sess}")
        subj_short = subj.split("-")[-1]
        task_short = task.split("-")[-1]
        out_path = os.path.join(
            subj_deriv_func,
            f"{subj}_{sess}_{task}_name-{model_name}_"
            + f"level-{model_level}_betas.tsv",
        )
        if os.path.exists(out_path):
            return out_path

        #
        for self._design_path in design_list:
            run_dir = os.path.basename(os.path.dirname(self._design_path))
            run_num = run_dir.split("_")[0].split("-")[1]
            if len(run_num) != 2:
                raise ValueError("Error parsing path for run number")

            #
            cope_dict = self._find_copes()
            for emo, cope_path in cope_dict.items():
                print(f"\t\tExtracting betas for run-{run_num}: {emo}")
                h_arr = self.nifti_to_arr(cope_path)
                img_arr = self.add_arr_id(
                    subj_short, task_short, emo, h_arr, run=run_num
                )
                del h_arr

                # Create/update dataframe
                if "df_betas" not in locals() and "df_betas" not in globals():
                    df_betas = self.arr_to_df(img_arr)
                else:
                    df_tmp = self.arr_to_df(img_arr)
                    df_betas = pd.concat(
                        [df_betas, df_tmp], axis=0, ignore_index=True
                    )
                    del df_tmp
                del img_arr

        # Clean if workflow uses mask_coord
        print("\tCleaning dataframe ...")
        if hasattr(self, "_rm_cols"):
            df_betas = df_betas.drop(self._rm_cols, axis=1)

        # Write and clean
        df_betas.to_csv(out_path, index=False, sep="\t")
        print(f"\t\tWrote : {out_path}")
        del df_betas
        return out_path


def comb_matrices(subj_list, model_name, model_level, proj_deriv, out_dir):
    """Combine participant beta dataframes into master.

    Copied, lightly edits from func_model.resources.afni.group.comb_matrices,
    could be refactored into single method for AFNI, FSL.

    Find beta-coefficient dataframes for participants in subj_list
    and combine into a single dataframe. Output dataframe is written to:
        <out_dir>/fsl_<model_name>_<model_level>_betas.tsv

    Parameters
    ----------

    Returns
    -------
    path
        Location of output dataframe

    Raises
    ------
    ValueError
        Missing participant dataframes

    """
    print("\tCombining participant beta tsv files ...")

    # Find desired dataframes
    df_list = sorted(
        glob.glob(
            f"{proj_deriv}/model_fsl/sub*/ses*/func/*name-"
            + f"{model_name}_level-{model_level}_betas.tsv",
        )
    )
    if not df_list:
        raise ValueError("No subject beta-coefficient dataframes found.")
    beta_list = [
        x for x in df_list if os.path.basename(x).split("_")[0] in subj_list
    ]

    # Combine dataframes and write out
    for beta_path in beta_list:
        print(f"\t\tAdding {beta_path} ...")
        if "df_betas_all" not in locals() and "df_betas_all" not in globals():
            df_betas_all = pd.read_csv(beta_path, sep="\t")
        else:
            df_tmp = pd.read_csv(beta_path, sep="\t")
            df_betas_all = pd.concat(
                [df_betas_all, df_tmp], axis=0, ignore_index=True
            )

    out_path = os.path.join(
        out_dir, f"fsl_{model_name}_{model_level}_betas.tsv"
    )
    df_betas_all.to_csv(out_path, index=False, sep="\t")
    print(f"\tWrote : {out_path}")
    return out_path
