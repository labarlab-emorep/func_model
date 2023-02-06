"""Methods for group-level analyses."""
import os
import glob
import pandas as pd
import numpy as np
from func_model.resources.afni import helper
from func_model.resources.general import submit, matrix


class ExtractTaskBetas(matrix.NiftiArray):
    """Generate dataframe of voxel beta-coefficients.

    Split AFNI deconvolve files into desired sub-bricks, and then
    extract all voxel beta weights for sub-labels of interest.
    Convert extracted weights into a dataframe.

    Methods
    -------
    split_decon(emo_name=None)
        Split deconvolve file into each relevant sub-brick
    mask_coord()
        Find coordinates to censor based on brain mask
    make_func_matrix(subj, sess, task, model_name, decon_path)
        Generate dataframe of voxel beta weights for subject, session

    Example
    -------
    etb_obj = group.ExtractTaskBetas()
    etb_obj.mask_coord("/path/to/binary/mask/nii")
    df = etb_obj.make_func_matrix(*args)

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

    def _get_labels(self):
        """Get sub-brick levels from AFNI deconvolved file.

        Attributes
        ----------
        _stim_label : list
            Full label ID of sub-brick starting with "mov" or "sce",
            e.g. movFea#0_Coef.

        Raises
        ------
        ValueError
            Trouble parsing output of 3dinfo -label to list

        """
        print(f"\tGetting sub-brick labels for {self.subj}, {self.sess}")

        # Extract sub-brick label info
        out_label = os.path.join(self.subj_out_dir, "tmp_labels.txt")
        if not os.path.exists(out_label):
            bash_list = [
                "3dinfo",
                "-label",
                self.decon_path,
                f"> {out_label}",
            ]
            bash_cmd = " ".join(bash_list)
            _ = submit.submit_subprocess(bash_cmd, out_label, "Get labels")

        with open(out_label, "r") as lf:
            label_str = lf.read()
        label_list = [x for x in label_str.split("|")]
        if label_list[0] != "Full_Fstat":
            raise ValueError("Error in extracting decon labels.")

        # Identify labels relevant to task
        self._stim_label = [
            x
            for x in label_list
            if self.task.split("-")[1][:3] in x and "Fstat" not in x
        ]
        self._stim_label.sort()

    def split_decon(self, emo_name=None):
        """Split deconvolved files into files by sub-brick.

        Parameters
        ----------
        emo_name: str, optional
            Long name of emotion, for restricting beta extraction
            to single emotion

        Attributes
        ----------
        _beta_dict : dict
            key = emotion, value = path to sub-brick file

        Raises
        ------
        ValueError
            Sub-brick identifier int has length > 2

        """
        # Invert emo_switch to unpack sub-bricks
        _emo_switch = helper.emo_switch()
        emo_switch = {j: i for i, j in _emo_switch.items()}

        # Extract desired sub-bricks from deconvolve file by label name
        self._get_labels()
        beta_dict = {}
        for sub_label in self._stim_label:

            # Identify emo string, setup file name
            emo_long = emo_switch[sub_label[3:6]]
            if emo_name and emo_long != emo_name:
                continue
            out_file = (
                f"tmp_{self.subj}_{self.sess}_{self.task}_"
                + f"desc-{emo_long}_beta.nii.gz"
            )
            out_path = os.path.join(self.subj_out_dir, out_file)
            if os.path.exists(out_path):
                beta_dict[emo_long] = out_path
                continue

            # Determine sub-brick integer value
            print(f"\t\tExtracting sub-brick for {emo_long}")
            out_label = os.path.join(self.subj_out_dir, "tmp_label_int.txt")
            bash_list = [
                "3dinfo",
                "-label2index",
                sub_label,
                self.decon_path,
                f"> {out_label}",
            ]
            bash_cmd = " ".join(bash_list)
            _ = submit.submit_subprocess(bash_cmd, out_label, "Label int")

            with open(out_label, "r") as lf:
                label_int = lf.read().strip()
            if len(label_int) > 2:
                raise ValueError(f"Unexpected int length for {sub_label}")

            # Write sub-brick as new file
            bash_list = [
                "3dTcat",
                f"-prefix {out_path}",
                f"{self.decon_path}[{label_int}]",
                "> /dev/null 2>&1",
            ]
            bash_cmd = " ".join(bash_list)
            _ = submit.submit_subprocess(bash_cmd, out_label, "Split decon")
            beta_dict[emo_long] = out_path
        self._beta_dict = beta_dict

    def mask_coord(self, mask_path):
        """Identify censoring coordinates from binary brain mask.

        Read-in binary values from a brain mask, vectorize, and identify
        coordinates of mask file outside of brain. Sets internal attribute
        holding coordinates to remove from beta dataframes.

        Parameters
        ----------
        mask_path : path
            Location of binary brain mask

        Attributes
        ----------
        _rm_cols : array
            Column names (coordinates) to drop from beta dataframes

        """
        print("\tFinding coordinates to censor ...")
        img_flat = self.nifti_to_arr(mask_path)
        df_mask = self.arr_to_df(img_flat)
        self._rm_cols = df_mask.columns[df_mask.isin([0.0]).any()]

    def make_func_matrix(self, subj, sess, task, model_name, decon_path):
        """Generate a matrix of beta-coefficients from a deconvolve file.

        Extract task-related beta-coefficients maps, then vectorize the matrix
        and create a dataframe where each row has all voxel beta-coefficients
        for an event of interest.

        Dataframe is written to directory of <decon_path> and titled:
            <subj>_<sess>_<task>_desc-<model_name>_betas.tsv

        Parameters
        ----------
        subj : str
            BIDS subject identifier
        sess : str
            BIDS session identifier
        task : str
            [task-scenarios | task-movies]
            BIDS task identifier
        model_name : str
            Model identifier for deconvolution
        decon_path : path
            Location of deconvolved file

        Attributes
        ----------
        subj_out_dir : path
            Location for output files

        Returns
        -------
        path
            Location of output dataframe

        """

        def _id_arr(emo: str) -> np.ndarray:
            """Return array of identifier information."""
            return np.array(
                [
                    ["subj_id", "task_id", "emo_id"],
                    [subj.split("-")[-1], task.split("-")[-1], emo],
                ]
            )

        # Check input, set attributes and output location
        if task not in ["task-scenarios", "task-movies"]:
            raise ValueError(f"Unexpected value for task : {task}")

        print(f"\tGetting betas from {subj}, {sess}")
        self.subj = subj
        self.sess = sess
        self.task = task
        self.decon_path = decon_path
        self.subj_out_dir = os.path.dirname(decon_path)
        out_path = os.path.join(
            self.subj_out_dir,
            f"{subj}_{sess}_{task}_desc-{model_name}_betas.tsv",
        )
        if os.path.exists(out_path):
            return out_path

        # Generate/Get seprated decon behavior files
        self.split_decon()

        # Extract and vectorize voxel betas
        for emo, beta_path in self._beta_dict.items():
            print(f"\t\tExtracting betas for : {emo}")
            img_arr = self.nifti_to_arr(beta_path)
            info_arr = _id_arr(emo)
            img_arr = np.concatenate((info_arr, img_arr), axis=1)

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
        tmp_list = glob.glob(f"{self.subj_out_dir}/tmp_*")
        for rm_file in tmp_list:
            os.remove(rm_file)
        return out_path


def comb_matrices(subj_list, model_name, proj_deriv, out_dir):
    """Combine participant beta dataframes into master.

    Find beta-coefficient dataframes for participants in subj_list
    and combine into a single dataframe.

    Output dataframe is written to:
        <out_dir>/afni_<model_name>_betas.tsv

    Parameters
    ----------
    subj_list : list
        Participants to include in final dataframe
    model_name : str
        Model identifier for deconvolution
    proj_deriv : path
        Location of project derivatives, will search for dataframes
        in <proj_deriv>/model_afni/sub-*.

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
            f"{proj_deriv}/model_afni/**/*desc-{model_name}_betas.tsv",
            recursive=True,
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

    out_path = os.path.join(out_dir, f"afni_{model_name}_betas.tsv")
    df_betas_all.to_csv(out_path, index=False, sep="\t")
    print(f"\tWrote : {out_path}")
    return out_path
