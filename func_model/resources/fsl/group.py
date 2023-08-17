"""Methods for group-level analyses.

ExtractTaskBetas : mine nii to generate dataframes of beta estimates
comb_matrices : concatenate participant beta-estimate dfs
ImportanceMask : generate mask in template space from classifier output

"""
import os
import glob
import re
import json
import pandas as pd
import numpy as np
from multiprocessing import Pool
import nibabel as nib
from func_model.resources.fsl import helper
from func_model.resources.general import matrix


class ExtractTaskBetas(matrix.NiftiArray):
    """Generate dataframe of voxel beta-coefficients.

    Align FSL cope.nii files with their corresponding contrast
    names and then extract all voxel beta weights for contrasts
    of interest. Converts extracted beta weights into a dataframe.

    Inherits general.matrix.NiftiArray.

    Parameters
    ----------
    proj_dir : path
        Location of project directory

    Methods
    -------
    make_func_matrix(**args)
        Identify and align cope.nii files, mine for betas
        and generate dataframe

    Example
    -------
    etb_obj = group.ExtractTaskBetas(*args)
    etb_obj.mask_coord("/path/to/binary/mask/nii")
    df_path = etb_obj.make_func_matrix(*args)

    """

    def __init__(self, proj_dir):
        """Initialize."""
        print("Initializing ExtractTaskBetas")
        super().__init__()
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

    def _drop_contrast(self):
        """Drop contrasts of no interest from self._con_dict."""
        if not hasattr(self, "_con_dict"):
            raise AttributeError(
                "Missing self._con_dict, try self._read_contrast"
            )
        for key in list(self._con_dict.keys()):
            if not key.startswith(self._con_name):
                del self._con_dict[key]

    def _clean_contrast(self):
        """Clean self._con_dict into pairs of emotion: contrast num."""
        if not hasattr(self, "_con_dict"):
            raise AttributeError(
                "Missing self._con_dict, try self._read_contrast"
            )
        out_dict = {}
        clean_num = len(self._con_name)
        for key, value in self._con_dict.items():
            new_key = key[clean_num:].split("GT")[0].lower()
            out_dict[new_key] = value[-1]
        self._con_dict = out_dict

    def _find_copes(self) -> dict:
        """Match emotion name to cope file."""
        # Mine, organize design.con info
        self._read_contrast()
        self._drop_contrast()
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
        con_name,
        design_list,
        subj_out,
        mot_thresh=0.2,
        overwrite=False,
    ):
        """Generate a matrix of beta-coefficients from FSL GLM cope files.

        Extract task-related beta-coefficients maps, then vectorize the matrix
        and create a dataframe where each row has all voxel beta-coefficients
        for an event of interest.

        Dataframe is written to:
            <out_dir>/<subj>_<sess>_<task>_<level>_<name>_<contrast>_betas.tsv

        Parameters
        ----------
        subj : str
            BIDS subject identifier
        sess : str
            BIDS session identifier
        task : str
            [task-movies | task-scenarios]
            BIDS task identifier
        model_name : str
            [sep | tog]
            FSL model identifier
        model_level : str
            [first]
            FSL model level
        con_name : str
            [stim | replay | tog]
            Desired contrast from which coefficients will be extracted
        design_list : list
            Paths to participant design.con files
        subj_out : path
            Output location for betas dataframe
        mot_thresh : float, optional
            Runs with a proportion of volumes >= mot_thresh will
            not be included in output dataframe
        overwrite : bool, optional
            Whether to overwrite existing beta TSV files

        Returns
        -------
        path
            Output location of beta dataframe

        Raises
        ------
        FileNotFoundError
            Missing outlier proportion file
        ValueError
            Unexpected task, model_name, model_level
            Trouble deriving run number

        """
        # Validate model variables
        if not helper.valid_task(task):
            raise ValueError(f"Unexpected value for task : {task}")
        if model_name not in ["sep", "tog"]:
            raise ValueError(
                f"Unsupported value for model_name : {model_name}"
            )
        if not helper.valid_level(model_level):
            raise ValueError(
                f"Unsupported value for model_level : {model_level}"
            )
        if not helper.valid_contrast(con_name):
            raise ValueError(f"Unsupported value for con_name : {con_name}")
        if not isinstance(overwrite, bool):
            raise TypeError("Expected type bool for overwrite")

        # Setup and check for existing work
        print(f"\tGetting betas from {subj}, {sess}")
        out_path = os.path.join(
            subj_out,
            f"{subj}_{sess}_{task}_level-{model_level}_"
            + f"name-{model_name}_con-{con_name}_betas.tsv",
        )
        if os.path.exists(out_path) and not overwrite:
            return out_path
        self._con_name = con_name

        # Mine files from each design.con, run in parallel
        mult_df = Pool(processes=8).starmap(
            self._mine_copes,
            [
                (
                    design_path,
                    subj,
                    sess,
                    task,
                    subj_out,
                    mot_thresh,
                )
                for design_path in design_list
            ],
        )
        df_betas = pd.concat(mult_df, axis=0, ignore_index=True)
        del mult_df

        # Clean if workflow uses mask_coord
        print("\tCleaning dataframe ...")
        if hasattr(self, "_rm_cols"):
            df_betas = df_betas.drop(self._rm_cols, axis=1)

        # Write and clean
        df_betas.to_csv(out_path, index=False, sep="\t")
        print(f"\t\tWrote : {out_path}")
        del df_betas
        return out_path

    def _mine_copes(
        self,
        design_path,
        subj,
        sess,
        task,
        subj_out,
        mot_thresh,
    ):
        """Vectorize cope betas for run."""
        self._design_path = design_path
        subj_short = subj.split("-")[-1]
        task_short = task.split("-")[-1]

        # Determine run number for identifier columns
        run_dir = os.path.basename(os.path.dirname(self._design_path))
        run_num = run_dir.split("_")[0].split("-")[1]
        if len(run_num) != 2:
            raise ValueError("Error parsing path for run number")

        # Compare proportion of outliers to criterion, skip run
        # if the the threshold is exceeded
        prop_path = os.path.join(
            subj_out,
            "confounds_proportions",
            f"{subj}_{sess}_{task}_run-{run_num}_"
            + "desc-confounds_proportion.json",
        )
        if not os.path.exists(prop_path):
            raise FileNotFoundError(f"Expected to find : {prop_path}")
        with open(prop_path) as jf:
            prop_dict = json.load(jf)
        prop_mot = prop_dict["CensorProp"]
        if prop_mot >= mot_thresh:
            return

        # Find and match copes to emotions, get voxel betas
        cope_dict = self._find_copes()
        for emo, cope_path in cope_dict.items():
            h_arr = self.nifti_to_arr(cope_path)
            img_arr = self.add_arr_id(
                subj_short,
                task_short,
                emo,
                h_arr,
                run=run_num,
            )

            # Create/update dataframe
            if "df_out" not in locals():
                df_out = self.arr_to_df(img_arr)
            else:
                df_tmp = self.arr_to_df(img_arr)
                df_out = pd.concat(
                    [df_out, df_tmp],
                    axis=0,
                    ignore_index=True,
                )
        return df_out


def comb_matrices(
    subj_list, model_name, model_level, con_name, proj_deriv, out_dir
):
    """Combine participant beta dataframes into master.

    Find beta-coefficient dataframes for participants in subj_list
    and combine into a single dataframe.

    Parameters
    ----------
    subj_list : list
        Participants to include in final dataframe
    model_name : str
        [sep]
        FSL model identifier
    model_level : str
        [first]
        FSL model level
    con_name : str
        [stim | replay]
        Desired contrast from which coefficients will be extracted
    proj_deriv : path
        Location of project derivatives, will search for dataframes
        in <proj_deriv>/model_fsl/sub-*.
    out_dir : path
        Output location of final dataframe

    Returns
    -------
    tuple
        [0] = pd.DataFrame
        [1] = str, os.PathLike
            Location of output dataframe

    Raises
    ------
    ValueError
        Unexpected values for model_name, model_level
        Missing participant dataframes

    """
    if not helper.valid_name(model_name):
        raise ValueError(f"Unsupported value for model_name : {model_name}")
    if not helper.valid_level(model_level):
        raise ValueError(f"Unsupported value for model_level : {model_level}")
    if not helper.valid_contrast(con_name):
        raise ValueError(f"Unsupported value for con_name : {con_name}")

    # Find desired dataframes
    print("\tCombining participant beta tsv files ...")
    df_list = sorted(
        glob.glob(
            f"{proj_deriv}/model_fsl/sub*/ses*/func/*level-{model_level}_"
            + f"name-{model_name}_con-{con_name}_betas.tsv",
        )
    )
    if not df_list:
        raise ValueError("No subject beta-coefficient dataframes found.")
    beta_list = [
        x for x in df_list if os.path.basename(x).split("_")[0] in subj_list
    ]

    # Combine all beta TSVs, load data in parallel
    all_betas = Pool().starmap(
        helper.load_tsv, [(beta_path,) for beta_path in beta_list]
    )
    df_betas_all = pd.concat(all_betas, axis=0, ignore_index=True)
    out_path = os.path.join(
        out_dir,
        f"level-{model_level}_name-{model_name}_con-{con_name}Washout_"
        + "voxel-betas.tsv",
    )
    df_betas_all.to_csv(out_path, index=False, sep="\t")
    print(f"\tWrote : {out_path}")
    return (df_betas_all, out_path)


# %%
class ImportanceMask(matrix.NiftiArray):
    """Convert a dataframe of classifier values into a NIfTI mask.

    Reference a template to derive header information and start a
    matrix of the same size. Populate said matrix was row values
    from a supplied dataframe.

    Inherits general.matrix.NiftiArray.

    Methods
    -------
    mine_template(tpl_path)
        Extract relevant information from a template
    make_mask(df, mask_path)
        Turn dataframe of values into NIfTI mask

    Example
    -------
    im_obj = group.ImportanceMask()
    im_obj.mine_template("/path/to/template.nii.gz")
    im_obj.make_mask(pd.DataFrame, "/path/to/output/mask.nii.gz")

    """

    def __init__(self):
        """Initialize."""
        print("Initializing ImportanceMask")
        super().__init__()

    def mine_template(self, tpl_path):
        """Mine a NIfTI template for information.

        Capture the NIfTI header and generate an empty
        np.ndarray of the same size as the template.

        Parameters
        ----------
        tpl_path : path
            Location and name of template

        Attributes
        ----------
        img_header : obj, nibabel.nifti1.Nifti1Header
            Header data of template
        empty_matrix : np.ndarray
            Matrix containing zeros that is the same size as
            the template.

        """
        if not os.path.exists(tpl_path):
            raise FileNotFoundError(f"Missing file : {tpl_path}")

        print(f"\tMining domain info from : {tpl_path}")
        img = self.nifti_to_img(tpl_path)
        img_data = img.get_fdata()
        self.img_header = img.header
        self.empty_matrix = np.zeros(img_data.shape)

    def make_mask(self, df, mask_path):
        """Convert row values into matrix and save as NIfTI mask.

        Using the dataframe column names, fill an empty matrix
        with row values and then save the file as a NIfTI mask.

        Parameters
        ----------
        df : pd.DataFrame
            A header and single row containing classifier importance.
            Column names should be formatted as coordinate, e.g.
            "(45, 31, 90)".
        mask_path : path
            Location and name of output NIfTI file

        Returns
        -------
        nd.array
            Matrix of template size filled with classifier
            importance values.

        Raises
        ------
        AttributeError
            Missing required attributes
        KeyError
        ValueError
            Improper formatting of dataframe

        """
        # Check for required attrs
        if not hasattr(self, "empty_matrix") and not hasattr(
            self, "img_header"
        ):
            raise AttributeError(
                "Attributes empty_matrix, img_header "
                + "required. Try ImportanceMask.mine_template."
            )

        # Validate dataframe
        if df.shape[0] != 1:
            raise ValueError("Dataframe must have only one row")
        chk_col = df.columns[0]
        try:
            int(re.sub("[^0-9]", "", chk_col))
        except ValueError:
            raise KeyError("Improperly formatted df column name.")
        if len(re.sub("[^0-9]", " ", chk_col).split()) != 3:
            raise KeyError("Improperly formatted df column name.")

        # Convert column names into a list of coordinate values
        print(f"\tBuilding importance map : {mask_path}")
        arr_fill = self.empty_matrix.copy()
        col_emo = [re.sub("[^0-9]", " ", x).split() for x in df.columns]

        # Add each column's value to the appropriate coordinate
        # in the empty matrix.
        for col_idx in col_emo:
            x = int(col_idx[0])
            y = int(col_idx[1])
            z = int(col_idx[2])
            arr_fill[x][y][z] = df.loc[0, f"({x}, {y}, {z})"]

        # Write matrix as a nii, embed template header
        emo_img = nib.Nifti1Image(
            arr_fill, affine=None, header=self.img_header
        )
        nib.save(emo_img, mask_path)
        return arr_fill
