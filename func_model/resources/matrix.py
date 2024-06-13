"""Resources dealing with NIfTI matrices.

NiftiArray : manage converting nii voxel values to arrays
C3dMethods : useful methods using c3d for nii manipulation

"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from func_model.resources import submit


class NiftiArray:
    """Convert NIfTI to dataframe.

    Helper methods for converting 3D NIfTI files to 1D arrays
    and dataframes.

    Parameters
    ----------
    float_prec : int, optional
        Float precision of dataframe

    Attributes
    ----------
    rm_voxels : list
        Which voxels have only 0 in mask NIfTI, set
        by mask_coord

    Methods
    -------
    arr_to_df(arr)
        Convert 1D array to pd.DataFrame
    mask_coord(mask_path)
        Identify coordinates outside of group-level binary mask,
        sets rm_voxels attr
    nifti_to_arr(nifti_path)
        Convert 3D NIfTI to 1D array
    nifti_to_img(nifti_path)
        Convert NIfTI to Nibabel image

    Example
    -------
    na_obj = matrix.NiftiArray(4)
    na_obj.mask_coord("/path/to/mask/nii")
    img_flat = na_obj.nifti_to_arr("/path/to/nii")
    id_flat = na_obj.add_arr_id("ER0009", "movies", "fear", img_flat)
    df_flat = na_obj.arr_to_df(id_flat)

    """

    def __init__(self, float_prec: int = 4):
        """Initialize."""
        if not isinstance(float_prec, int):
            raise TypeError("Expected float_prec type int")
        print("\tInitializing NiftiArray")
        self._float_prec = float_prec

    def _flatten_array(self, arr: np.ndarray) -> np.ndarray:
        """Flatten 3D array and keep xyz index."""
        idx_val = []
        for x in np.arange(arr.shape[0]):
            for y in np.arange(arr.shape[1]):
                for z in np.arange(arr.shape[2]):
                    idx_val.append(
                        [
                            f"{x}.{y}.{z}",
                            round(arr[x][y][z], self._float_prec),
                        ]
                    )
        idx_val_arr = np.array(idx_val, dtype=object)
        return np.transpose(idx_val_arr)

    def nifti_to_arr(self, nifti_path: str) -> np.ndarray:
        """Generate flat array of NIfTI voxel values."""
        img = self.nifti_to_img(nifti_path)
        img_data = img.get_fdata()
        img_flat = self._flatten_array(img_data)
        return img_flat

    def nifti_to_img(self, nifti_path: str):
        """Return Nibabel Image."""
        return nib.load(nifti_path)

    def arr_to_df(self, arr: np.ndarray, col_name: str) -> pd.DataFrame:
        """Make dataframe from flat array, drop indices in rm_voxels attr."""
        df = pd.DataFrame(np.transpose(arr), columns=["voxel_name", col_name])
        df = df.set_index("voxel_name")
        if hasattr(self, "rm_voxels"):
            df = df.drop(self.rm_voxels)
        return df

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
        rm_voxels : list
            Indices (voxel coordinates) to drop from beta dataframes

        """
        print("\tFinding coordinates to censor ...")
        img_flat = self.nifti_to_arr(mask_path)
        df_mask = self.arr_to_df(img_flat, "voxel_value")
        self.rm_voxels = df_mask.index[df_mask["voxel_value"] == 0.0].to_list()


class C3dMethods:
    """Various c3d-based methods to help processing NIfTI files.

    Parameters
    ----------
    out_dir : path
        Output location

    Methods
    -------
    thresh(*args)
        Threshold voxels values between a range
    comb(*args)
        Combine a stack of images by summation
    mult(*args)
        Multiply to images

    Examples
    --------
    c3d_meth = helper.C3dMethods(out_dir)
    out_thresh = c3d_meth.thresh(1, 3, 1, 0, foo.nii, "binary")
    out_sum = c3d_meth.comb([foo.nii, bar.nii], "sum")
    out_mult = c3d_meth.mult(foo.nii bar.nii "multiply)

    """

    def __init__(self, out_dir):
        """Initialize."""
        self.out_dir = out_dir
        if "c3d" not in os.environ["PATH"]:
            raise EnvironmentError("c3d not found in OS PATH")

    def thresh(
        self,
        lb,
        ub,
        vin,
        vout,
        in_file,
        out_name,
    ):
        """Threshold labelled NIfTI.

        Replace values within a range with one value, and values
        outside of the range with another.

        Writes a file to:
            <out_dir>/<out_name>.nii.gz

        Parameters
        ----------
        lb : int
            Lower bound
        ub : int
            Upper bound
        vin : int
            Output value for voxels within range
            (lb < vox < ub)
        vout : int
            Output value for voxels outside range
            (vox < lb & ub < vox)
        in_file : path
            Input nii file
        out_name : str
            Output name

        Returns
        -------
        path
            Location of output file

        """
        out_path = os.path.join(self.out_dir, f"{out_name}.nii.gz")
        c3d_cmd = f"""
            c3d \
                {in_file} \
                -thresh {lb} {ub} {vin} {vout} \
                -o {out_path}
        """
        submit.submit_subprocess(c3d_cmd, out_path, "Thresh")
        return out_path

    def comb(self, in_files, out_name):
        """Combine multiple NIfTI masks.

        Iteratively sum a list of files (masks) to combine
        all files.

        Writes a file to:
            <out_dir>/<out_name>.nii.gz

        Parameters
        ----------
        in_files : list
            Paths to multiple masks
        out_name : str
            Output name

        Returns
        -------
        path
            Location of output file

        """
        out_path = os.path.join(self.out_dir, f"{out_name}.nii.gz")
        in_list = " ".join(in_files)
        c3d_cmd = f"""
            c3d \
                {in_list} \
                -accum -add -endaccum \
                -o {out_path}
        """
        submit.submit_subprocess(c3d_cmd, out_path, "Combine")
        return out_path

    def mult(self, in_a, in_b, out_name):
        """Multiply two NIfTI files.

        Writes a file to:
            <out_dir>/<out_name>.nii.gz

        Parameters
        ----------
        in_a : path
            Location of file A
        in_b : path
            Location of file B
        out_name : str
            Output name

        Returns
        -------
        path
            Location of output file

        """
        out_path = os.path.join(self.out_dir, f"{out_name}.nii.gz")
        c3d_cmd = f"""
            c3d \
                {in_a} {in_b} \
                -multiply \
                -o {out_path}
        """
        submit.submit_subprocess(c3d_cmd, out_path, "Combine")
        return out_path
