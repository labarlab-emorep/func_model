"""Methods for group-level analyses.

ExtractTaskBetas : mine nii to generate dataframes of beta estimates
ImportanceMask : generate mask in template space from classifier output
ConjunctAnalysis : generate conjunction maps from ImportanceMask output

"""
import os
import re
import json
from typing import Union, Tuple
import pandas as pd
import numpy as np
from multiprocessing import Pool
import nibabel as nib
from func_model.resources.fsl import helper as fsl_helper
from func_model.resources.general import matrix
from func_model.resources.general import submit
from func_model.resources.general import sql_database


class _GetCopes:
    """Title."""

    def __init__(self, con_name):
        """Title."""
        self._con_name = con_name  # used for cleaning design.con strings

    def find_copes(self, design_path: Union[str, os.PathLike]) -> dict:
        """Match emotion name to cope file."""
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
        for key in list(self._con_dict.keys()):
            if not key.startswith(self._con_name):
                del self._con_dict[key]

    def _clean_contrast(self):
        """Clean self._con_dict into pairs of emotion: contrast num."""
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

    Inherits general.matrix.NiftiArray.

    Methods
    -------
    make_func_matrix(*args)
        Identify and align cope.nii files, mine for betas
        and generate dataframe

    Example
    -------
    etb_obj = group.ExtractTaskBetas()
    etb_obj.mask_coord("/path/to/binary/mask.nii")
    df_path = etb_obj.make_func_matrix(*args)

    """

    def __init__(self):
        """Initialize."""
        print("Initializing ExtractTaskBetas")
        super().__init__()

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

        Returns
        -------
        list
            Output locations of beta dataframe

        Raises
        ------
        FileNotFoundError
            Missing outlier proportion file
        ValueError
            Unexpected task, model_name, model_level
            Trouble deriving run number

        """
        # Validate model variables
        if not fsl_helper.valid_task(task):
            raise ValueError(f"Unexpected value for task : {task}")
        if model_name not in ["sep", "tog", "lss"]:
            raise ValueError(
                f"Unsupported value for model_name : {model_name}"
            )
        if not fsl_helper.valid_level(model_level):
            raise ValueError(
                f"Unsupported value for model_level : {model_level}"
            )
        if not fsl_helper.valid_contrast(con_name):
            raise ValueError(f"Unsupported value for con_name : {con_name}")

        # Setup and check for existing work
        self._subj = subj
        self._sess = sess
        self._task = task
        self._model_name = f"name-{model_name}"
        self._model_level = f"level-{model_level}"
        self._con_name = con_name
        self._subj_out = subj_out
        self._mot_thresh = mot_thresh

        #
        self._get_copes = _GetCopes(con_name)

        # Mine files from each design.con, run in parallel
        self._data_obj = Pool(processes=8).starmap(
            self._mine_copes,
            [(design_path,) for design_path in design_list],
        )

        #
        self._write_csv()
        self._update_mysql()

    def _mine_copes(
        self,
        design_path: Union[str, os.PathLike],
    ) -> Tuple:
        """Vectorize cope betas for run."""
        # Determine run number for file name
        _run_dir = os.path.basename(os.path.dirname(design_path))
        run_num = _run_dir.split("_")[0].split("-")[1]
        run = f"run-{run_num}"
        if len(run_num) != 2:
            raise ValueError("Error parsing path for run number")

        # Compare proportion of outliers to criterion, skip run
        # if the the threshold is exceeded
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
        print(f"\tGetting betas from {self._subj}, {self._sess}, {run}")
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
        """Title."""
        for idx, _ in enumerate(self._data_obj):
            df = self._data_obj[idx][0]
            run = self._data_obj[idx][1]

            #
            out_path = os.path.join(
                self._subj_out,
                f"{self._subj}_{self._sess}_{self._task}_run-0{run}_"
                + f"{self._model_level}_{self._model_name}_"
                + f"con-{self._con_name}_betas.csv",
            )
            print(f"\tWriting : {out_path}")
            df.to_csv(out_path, index=False)

    def _update_mysql(self):
        """Title."""
        #
        df_a = self._data_obj[0][0][["voxel_name"]].copy()
        df_b = self._data_obj[0][0][["voxel_name"]].copy()
        df_a["num_exposure"] = 1
        df_b["num_exposure"] = 2

        #
        for idx, _ in enumerate(self._data_obj):
            df = self._data_obj[idx][0]
            run_num = self._data_obj[idx][1]
            if run_num < 5:
                df_a = df_a.merge(df, how="left", on="voxel_name")
            else:
                df_b = df_b.merge(df, how="left", on="voxel_name")
        df_in = pd.concat([df_a, df_b])
        del self._data_obj, df_a, df_b

        #
        db_con = sql_database.DbConnect()
        up_mysql = sql_database.MysqlUpdate(db_con)
        up_mysql.update_db(
            df_in,
            self._subj,
            self._task,
            self._model_name.split("-")[-1],
            self._con_name,
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
