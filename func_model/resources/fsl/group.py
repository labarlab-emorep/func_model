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
        """Title."""
        con_lines = []
        with open(design_path) as dp:
            for ln in dp:
                if ln.startswith("/ContrastName"):
                    con_lines.append(ln[1:])
        con_dict = {}
        for line in con_lines:
            con, name = line.split()
            con_dict[name] = con

    def _drop_replay(self):
        """Title."""
        for key in list(con_dict.keys()):
            if key.startswith("replay"):
                del con_dict[key]

    def _clean_contrast(self):
        """Title."""
        out_dict = {}
        for key, value in con_dict.items():
            new_key = key[4:].split("GT")[0].lower()
            out_dict[new_key] = value[-1]
        con_dict = out_dict

    def _find_copes(self):
        """Title."""
        self._read_contrast()
        self._drop_replay()
        self._clean_contrast()

        #
        feat_dir = os.path.dirname(design_path)
        stats_dir = os.path.join(feat_dir, "stats")
        cope_dict = {}
        for emo, num in con_dict.items():
            nii_name = f"cope{num}.nii.gz"
            nii_path = os.path.join(stats_dir, nii_name)
            cope_dict[emo] = nii_path
        return cope_dict

    def make_func_matrix(
        self, subj, sess, task, model_name, model_level, design_list, out_dir
    ):
        """Title."""
        # Check input, set attributes and output location
        if not helper.valid_task(task):
            raise ValueError(f"Unexpected value for task : {task}")

        print(f"\tGetting betas from {subj}, {sess}")
        self.subj = subj
        self.sess = sess
        self.task = task
        subj_short = subj.split("-")[-1]
        task_short = task.split("-")[-1]

        out_path = os.path.join(
            out_dir,
            f"{subj}_{sess}_{task}_name-{model_name}_"
            + f"level-{model_level}_betas.tsv",
        )
        if os.path.exists(out_path):
            return out_path

        #
        for design_path in design_list:
            cope_dict = self._find_copes()
            for emo, cope_path in cope_dict.items():
                print(f"\t\tExtracting betas for : {emo}")
                h_arr = self.nifti_to_arr(cope_path)
                img_arr = self.add_arr_id(subj_short, task_short, emo, h_arr)
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

        # # Clean if workflow uses mask_coord
        # print("\tCleaning dataframe ...")
        # if hasattr(self, "_rm_cols"):
        #     df_betas = df_betas.drop(self._rm_cols, axis=1)

        # # Write and clean
        # df_betas.to_csv(out_path, index=False, sep="\t")
        # print(f"\t\tWrote : {out_path}")
        # del df_betas
        # tmp_list = glob.glob(f"{self.subj_out_dir}/tmp_*")
        # for rm_file in tmp_list:
        #     os.remove(rm_file)
        # return out_path
