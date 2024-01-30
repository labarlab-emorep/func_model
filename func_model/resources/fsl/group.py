"""Methods for group-level analyses.

ExtractTaskBetas : mine nii to generate dataframes of beta estimates
ImportanceMask : generate mask in template space from classifier output
ConjunctAnalysis : generate conjunction maps from ImportanceMask output

"""
import os
import re
import json
from typing import Union, Tuple
import numpy as np
import pandas as pd
from multiprocessing import Pool
import nibabel as nib
from func_model.resources.fsl import helper as fsl_helper
from func_model.resources.general import matrix
from func_model.resources.general import submit
from func_model.resources.general import sql_database


class _GetCopes:
    """Mine design.con to return {emo: nii} dict.

    Methods
    -------
    find_copes()
        Find copes and return {emo: nii} mapping

    """

    def __init__(self, con_name: str):
        """Initialize."""
        self._con_name = con_name

    def find_copes(self, design_path: Union[str, os.PathLike]) -> dict:
        """Match emotion name to cope file, return {emo: nii}."""
        self._design_path = design_path

        # Mine, organize design.con info
        self._read_contrast()
        self._drop_contrast()
        self._clean_contrast()

        # Orient from design.con to stats dir
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

    def _read_contrast(self):
        """Match contrast name to number."""
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
        """Drop contrasts of no interest."""
        for key in list(self._con_dict.keys()):
            if not key.startswith(self._con_name):
                del self._con_dict[key]

    def _clean_contrast(self):
        """Clean emotion: contrast pairs."""
        out_dict = {}
        clean_num = len(self._con_name)
        for key, value in self._con_dict.items():
            new_key = key[clean_num:].split("GT")[0].lower()
            out_dict[new_key] = value[-1]
        self._con_dict = out_dict


class ExtractTaskBetas(matrix.NiftiArray):
    """Generate dataframe of voxel beta-coefficients.

    Align FSL cope.nii files with their corresponding contrast
    names and then extract all voxel beta weights for contrasts
    of interest. Converts extracted beta weights into a dataframe.

    Extracted beta weights are sent to a tbl_betas_* in db_emorep,
    and are also written to Keoki. Inherits
    func_model.resources.general.matrix.NiftiArray.

    Methods
    -------
    make_beta_matrix(*args)
        Identify and align cope.nii files, mine for betas
        and generate dataframe

    Example
    -------
    etb_obj = group.ExtractTaskBetas()
    etb_obj.mask_coord("/path/to/binary/mask.nii")
    etb_obj.make_beta_matrix(*args)

    """

    def __init__(self):
        """Initialize."""
        print("Initializing ExtractTaskBetas")
        super().__init__()

    def make_beta_matrix(
        self,
        subj,
        sess,
        task,
        model_name,
        model_level,
        con_name,
        design_list,
        subj_out,
        overwrite,
        mot_thresh=0.2,
    ):
        """Generate a matrix of beta-coefficients from FSL GLM cope files.

        Extract task-related beta-coefficients maps, then vectorize the matrix
        and create a dataframe where each row has all voxel beta-coefficients
        for an event of interest.

        Parameters
        ----------
        subj : str
            BIDS subject identifier
        sess : str
            BIDS session identifier
        task : str
            {"task-movies", "task-scenarios"}
            BIDS task identifier
        model_name : str
            {"lss", "sep"}
            FSL model identifier
        model_level : str
            {"first"}
            FSL model level
        con_name : str
            {"stim", "tog"}
            Desired contrast from which coefficients will be extracted
        design_list : list
            Paths to participant design.con files
        subj_out : path
            Output location for betas dataframe
        overwrite : bool
            Whether to overwrite existing data
        mot_thresh : float, optional
            Runs with a proportion of volumes >= mot_thresh will
            not be included in output dataframe

        Notes
        -----
        - model_name="sep" requires con_name="stim"
        - model_name="lss" requires con_name="tog"

        """
        # Validate args and setup
        if not fsl_helper.valid_task(task):
            raise ValueError(f"Unexpected value for task : {task}")
        if model_name not in ["sep", "lss"]:
            raise ValueError(
                f"Unsupported value for model_name : {model_name}"
            )
        if model_level not in ["first"]:
            raise ValueError(
                f"Unsupported value for model_level : {model_level}"
            )
        if con_name not in ["stim", "tog"]:
            raise ValueError(f"Unsupported value for con_name : {con_name}")
        if model_name == "sep" and con_name != "stim":
            raise ValueError(
                "Unexpected model contrast pair : "
                + f"{model_name}, {con_name}"
            )
        if model_name == "lss" and con_name != "tog":
            raise ValueError(
                "Unexpected model contrast pair : "
                + f"{model_name}, {con_name}"
            )

        self._subj = subj
        self._sess = sess
        self._task = task
        self._model_name = f"name-{model_name}"
        self._model_level = f"level-{model_level}"
        self._con_name = con_name
        self._subj_out = subj_out
        self._overwrite = overwrite
        self._mot_thresh = mot_thresh

        # Check if records already exist in db_emorep
        print(f"Working on {subj}, {task}, {model_name}, {con_name}")
        data_exist = self._check_exist()
        if not self._overwrite and data_exist:
            print(
                f"\tData already exist for {subj}, {task}, "
                + f"{model_name}, {con_name}; Continuing ..."
            )
            return

        # Orient to design.con files and mine niis in parallel
        self._get_copes = _GetCopes(con_name)
        with Pool(processes=8) as pool:
            items = [(x) for x in design_list]
            self._data_obj = pool.map(self._mine_copes, items)
        if not isinstance(self._data_obj[0], tuple):
            return

        # Write csv and update db_emorep
        # TODO deprecate _write_csv()
        if model_name != "lss":
            self._write_csv()

        if model_name == "lss":
            self._update_fsl_betas_lss()
        else:
            self._update_fsl_betas_reg()
        del self._data_obj

    def _check_exist(self) -> bool:
        """Return bool of whether record exists in db_emorep."""
        # Using separate connection here from _update_db_emorep
        # to avoid multiproc pickle issue.
        db_con = sql_database.DbConnect()
        update_betas = sql_database.DbUpdateBetas(db_con)
        data_exist = update_betas.check_db(
            self._subj,
            self._task,
            self._model_name.split("-")[-1],
            self._con_name,
        )
        db_con.close_con()
        return data_exist

    def _mine_copes(
        self,
        design_path: Union[str, os.PathLike],
    ) -> Tuple:
        """Vectorize cope betas, return tuple of pd.DataFrame, run number."""
        # Determine run number for file name
        _run_dir = os.path.basename(os.path.dirname(design_path))
        run_num = _run_dir.split("_")[0].split("-")[1]
        run = f"run-{run_num}"
        if len(run_num) != 2:
            raise ValueError("Error parsing path for run number")

        # Compare proportion of outliers to criterion, skip run
        # if the threshold is exceeded.
        prop_path = os.path.join(
            self._subj_out,
            "confounds_proportions",
            f"{self._subj}_{self._sess}_{self._task}_{run}_"
            + "desc-confounds_proportion.json",
        )
        if not os.path.exists(prop_path):
            raise FileNotFoundError(f"Expected to find : {prop_path}")
        with open(prop_path) as jf:
            prop_dict = json.load(jf)
        if prop_dict["CensorProp"] >= self._mot_thresh:
            return

        # Find and match copes to emotions, get voxel betas
        print(f"\tGetting betas from {self._subj}, {self._task}, {run}")
        cope_dict = self._get_copes.find_copes(design_path)
        for emo, cope_path in cope_dict.items():
            img_arr = self.nifti_to_arr(cope_path)

            # Create/update dataframe
            if "df_out" not in locals():
                df_out = self.arr_to_df(img_arr, f"emo_{emo}")
            else:
                df_tmp = self.arr_to_df(img_arr, f"emo_{emo}")
                df_out[f"emo_{emo}"] = df_tmp[f"emo_{emo}"]
                del df_tmp

        # Return df and run num (to ensure proper order from multiproc)
        df_out = df_out.reset_index()
        return (df_out, int(run_num))

    def _write_csv(self):
        """Write pd.DataFrame to disk."""
        # Unpack data_obj
        for idx, _ in enumerate(self._data_obj):
            if not isinstance(self._data_obj[idx], tuple):
                continue
            df = self._data_obj[idx][0]
            run = self._data_obj[idx][1]

            # Setup output path, write
            out_path = os.path.join(
                self._subj_out,
                f"{self._subj}_{self._sess}_{self._task}_run-0{run}_"
                + f"{self._model_level}_{self._model_name}_"
                + f"con-{self._con_name}_betas.csv",
            )
            print(f"\tWriting : {out_path}")
            df.to_csv(out_path, index=False)

    def _update_fsl_betas_reg(self):
        """Update db_emorep beta table for sep, tog models."""

        # Start left dfs of proper length, for each exposure
        df_a = self._data_obj[0][0][["voxel_name"]].copy()
        df_b = self._data_obj[0][0][["voxel_name"]].copy()
        df_a["num_exposure"] = 1
        df_b["num_exposure"] = 2

        # Unpack data_obj
        for idx, _ in enumerate(self._data_obj):
            if not isinstance(self._data_obj[idx], tuple):
                continue

            # Merge data with appropriate left df
            df = self._data_obj[idx][0]
            run_num = self._data_obj[idx][1]
            if run_num < 5:
                df_a = df_a.merge(df, how="left", on="voxel_name")
            else:
                df_b = df_b.merge(df, how="left", on="voxel_name")

        # Each proc gets own connection
        db_con = sql_database.DbConnect()
        update_betas = sql_database.DbUpdateBetas(db_con)

        # Update db_emorep beta table
        if df_a.shape[1] > 2:
            update_betas.update_db(
                df_a,
                self._subj,
                self._task,
                self._model_name.split("-")[-1],
                self._con_name,
                self._overwrite,
            )
        if df_b.shape[1] > 2:
            update_betas.update_db(
                df_b,
                self._subj,
                self._task,
                self._model_name.split("-")[-1],
                self._con_name,
                self._overwrite,
            )
        db_con.close_con()

    def _update_fsl_betas_lss(self):
        """Update db_emorep beta table for lss models."""

        def _mk_df() -> pd.DataFrame:
            """Return pd.DataFrame with voxel_name, num_event cols."""
            # Start with wide format
            df_par = self._data_obj[0][0][["voxel_name"]].copy()
            df_par["ev1"] = 1
            df_par["ev2"] = 2
            df_par["ev3"] = 3
            df_par["ev4"] = 4
            df_par["ev5"] = 5

            # Conver to long, clean redundant columns
            df = pd.wide_to_long(
                df_par, stubnames="ev", i="voxel_name", j="event"
            )
            df = df.reset_index()
            df = df.rename(columns={"event": "num_event"})
            df = df.drop(["ev"], axis=1)
            return df

        def _add_df(
            emo_name: str, df_l: pd.DataFrame, df_r: pd.DataFrame
        ) -> pd.DataFrame:
            """Update df_l with df_r data in col emo_name"""
            if emo_name not in df_l.columns:
                df_l = df_l.merge(
                    df_r,
                    how="left",
                    on=["voxel_name", "num_event"],
                )
            else:
                df_l = df_l.merge(
                    df_r,
                    how="left",
                    on=["voxel_name", "num_event"],
                )
                df_l[emo_name] = df_l[f"{emo_name}_y"].fillna(
                    df_l[f"{emo_name}_x"]
                )
                df_l = df_l.drop([f"{emo_name}_x", f"{emo_name}_y"], axis=1)
            return df_l

        # Start dfs for first, second emo block
        df_a = _mk_df()
        df_a["num_exposure"] = 1
        df_b = _mk_df()
        df_b["num_exposure"] = 2

        # Unpack data_obj
        for idx, _ in enumerate(self._data_obj):
            if not isinstance(self._data_obj[idx], tuple):
                continue
            df = self._data_obj[idx][0]
            run_num = self._data_obj[idx][1]

            # Identify emotion name and event number
            col_name = df.columns[1]
            emo_name = col_name.split("_ev")[0]
            ev_num = int(col_name[-1])

            # Prep df for merge, update parent dfs
            df = df.rename(columns={col_name: emo_name})
            df["num_event"] = ev_num
            if run_num < 5:
                df_a = _add_df(emo_name, df_a, df)
            else:
                df_b = _add_df(emo_name, df_b, df)

        # Convert missing data for SQL
        df_a = df_a.replace({np.nan: None})
        df_b = df_b.replace({np.nan: None})

        # Each proc gets own connection
        db_con = sql_database.DbConnect()
        update_betas = sql_database.DbUpdateBetas(db_con)

        # Update db_emorep beta table
        if df_a.shape[1] > 3:
            update_betas.update_db(
                df_a,
                self._subj,
                self._task,
                self._model_name.split("-")[-1],
                self._con_name,
                self._overwrite,
            )
        if df_b.shape[1] > 3:
            update_betas.update_db(
                df_b,
                self._subj,
                self._task,
                self._model_name.split("-")[-1],
                self._con_name,
                self._overwrite,
            )
        db_con.close_con()


# %%
class _MapMethods:
    """Voxel importance map methods.

    Methods
    -------
    c3d_add()
        Add voxels maps together
    cluster()
        Identify clusters of voxel value, size, and NN

    """

    def c3d_add(
        self,
        add_list: list,
        out_path: Union[str, os.PathLike],
    ):
        """Add 3D NIfTIs together."""
        bash_cmd = f"""\
            c3d \
                {" ".join(add_list)} \
                -accum -add -endaccum \
                -o {out_path}
        """
        _ = submit.submit_subprocess(bash_cmd, out_path, "c3d-add")
        self._tpl_head(out_path)

    def _tpl_head(self, out_path):
        """Fix header of conjunction files."""
        _ = submit.submit_subprocess(
            f"3drefit -space MNI {out_path}", out_path, "afni-refit"
        )

    def cluster(
        self,
        in_path: Union[str, os.PathLike],
        nn: int = 1,
        size: int = 10,
        vox_value: int = 2,
    ):
        """Identify clusters of NN, size, and voxel value."""
        out_dir = os.path.dirname(in_path)
        out_name = "Clust_" + os.path.basename(in_path)
        out_path = os.path.join(out_dir, out_name)
        bash_cmd = f"""\
            3dClusterize \
                -nosum -1Dformat \
                -inset {in_path} \
                -idat 0 -ithr 0 \
                -NN {nn} \
                -clust_nvox {size} \
                -bisided -{vox_value} {vox_value} \
                -pref_map {out_path} \
                > {out_path.replace(".nii.gz", ".txt")}
        """
        _ = submit.submit_subprocess(
            bash_cmd, out_path, "afni-clust", force_cont=True
        )


class ImportanceMask(matrix.NiftiArray, _MapMethods):
    """Convert a dataframe of classifier values into a NIfTI mask.

    Reference a template to derive header information and start a
    matrix of the same size. Populate said matrix was row values
    from a supplied dataframe.

    Inherits general.matrix.NiftiArray, _MapMethods

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
    im_obj.make_mask(
        pd.DataFrame, "/path/to/output/mask.nii.gz", "task-movies"
    )

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

    def make_mask(self, df, mask_path, task_name):
        """Convert row values into matrix and save as NIfTI mask.

        Using the dataframe column names, fill an empty matrix
        with row values and then save the file as a NIfTI mask.

        Parameters
        ----------
        df : pd.DataFrame
            A header and single row containing classifier importance.
            Column names should be formatted as coordinate, e.g.
            "(45, 31, 90)".
        mask_path : str, os.PathLike
            Location and name of output NIfTI file
        task_name : str
            Name of stimulus type

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
        clust_size = 5 if task_name == "scenarios" else 10
        self.cluster(mask_path, size=clust_size, vox_value=1)
        return arr_fill


class ConjunctAnalysis(_MapMethods):
    """Generate conjunction maps.

    Inherits _MapMethods.

    Generate omnibus, arousal, and valence conjunction maps
    from voxel importance maps.

    Parameters
    ----------
    map_list : list
        Paths to NIfTI voxel importance maps in template space
    out_dir : str, os.PathLike
        Output location

    Methods
    -------
    omni_map()
        Generate omnibus conjunction from all map_list files
    valence_map()
        Generate positive, negative, neutrual valence conjunction maps
    arousal_map()
        Generate high, medium, low arousal conjunction maps

    Example
    -------
    conj = fsl.group.ConjunctAnalysis(*args)
    conj.omni_map()
    conj.arousal_map()
    conj.valence_map()

    """

    def __init__(self, map_list, out_dir):
        """Initialize."""
        self._map_list = map_list
        self._out_dir = out_dir
        (
            self._model_level,
            self._model_name,
            self._task_name,
            self._con_name,
            _emo,
            _suff,
        ) = os.path.basename(map_list[0]).split("_")
        self._clust_size = 5 if self._task_name == "scenarios" else 10

    def omni_map(self):
        """Generate omnibus conjunction from all map_list files."""
        omni_out = os.path.join(
            self._out_dir,
            f"{self._model_level}_{self._model_name}_{self._task_name}_"
            + f"{self._con_name}_conj-omni_map.nii.gz",
        )
        print("Building conjunction map : omni")
        self.c3d_add(self._map_list, omni_out)
        self.cluster(omni_out, size=self._clust_size, vox_value=1)

    def valence_map(self):
        """Generate positive, negative, neutrual valence conjunction maps."""
        map_val = {
            "Pos": ["amusement", "awe", "excitement", "joy", "romance"],
            "Neg": [
                "anger",
                "anxiety",
                "disgust",
                "fear",
                "horror",
                "sadness",
            ],
            "Neu": ["calmness", "craving", "neutral", "surprise"],
        }
        self._conj_aro_val(map_val, "val")

    def _conj_aro_val(self, map_dict: dict, conj_name: str):
        """Unpack filenames to list items to make conj maps."""
        for key in map_dict:
            val_list = [
                x for x in self._map_list for y in map_dict[key] if y in x
            ]
            out_path = os.path.join(
                self._out_dir,
                f"{self._model_level}_{self._model_name}_{self._task_name}_"
                + f"{self._con_name}_conj-{conj_name}{key}_map.nii.gz",
            )
            print(f"Building conjunction map : {conj_name}{key}")
            self.c3d_add(val_list, out_path)
            self.cluster(out_path, size=self._clust_size, vox_value=1)

    def arousal_map(self):
        """Generate high, medium, low arousal conjunction maps."""
        map_aro = {
            "High": [
                "amusement",
                "anger",
                "anxiety",
                "craving",
                "disgust",
                "excitement",
                "fear",
                "horror",
                "surprise",
            ],
            "Low": ["calmness", "neutral", "sadness"],
            "Med": ["awe", "romance", "joy"],
        }
        self._conj_aro_val(map_aro, "aro")
