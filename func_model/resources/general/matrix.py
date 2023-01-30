"""Resource dealing with NIfTI matrices."""
import numpy as np
import pandas as pd
import nibabel as nib


class NiftiArray:
    """Convert NIfTI to dataframe.

    Helper methods for converting 3D NIfTI files to 1D arrays
    and dataframes.

    Methods
    -------
    arr_to_df(arr)
        Convert 1D array to pd.DataFrame
    nifti_to_arr(nifti_path)
        Convert 3D NIfTI to 1D array

    Example
    -------
    na_obj = matrix.NiftiArray(4)
    img_flat = na_obj.nifti_to_arr("/path/to/nii")
    df_flat = na_obj.arr_to_df(img_flat)

    """

    def __init__(self, float_prec: int):
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
                            f"({x}, {y}, {z})",
                            round(arr[x][y][z], self._float_prec),
                        ]
                    )
        idx_val_arr = np.array(idx_val, dtype=object)
        return np.transpose(idx_val_arr)

    def nifti_to_arr(self, nifti_path: str) -> np.ndarray:
        """Generate flat array of NIfTI voxel values."""
        img = nib.load(nifti_path)
        img_data = img.get_fdata()
        img_flat = self._flatten_array(img_data)
        return img_flat

    def arr_to_df(self, arr: np.ndarray) -> pd.DataFrame:
        """Make dataframe from flat array."""
        df = pd.DataFrame(np.transpose(arr), columns=["idx", "val"])
        df = df.set_index("idx")
        df = df.transpose().reset_index(drop=True)
        return df
