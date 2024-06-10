"""Methods for mask construction."""

import os
import glob
from func_model.resources.afni import helper as afni_helper
from func_model.resources.general import submit, matrix


class MakeMasks:
    """Generate masks for AFNI-style analyses.

    Make masks required by, suggested for AFNI-style deconvolutions
    and group analyses.

    Methods
    -------
    intersect()
        Generate an anatomical-functional intersection mask
    tissue()
        Make eroded tissue masks
    minimum()
        Mask voxels with some meaningful signal across all
        volumes and runs.

    """

    def __init__(
        self,
        subj_work,
        proj_deriv,
        anat_dict,
        func_dict,
    ):
        """Initialize object.

        Parameters
        ----------
        subj_work : path
            Location of working directory for intermediates
        proj_deriv : path
            Location of project derivatives, containing fmriprep
            and fsl_denoise sub-directories
        anat_dict : dict
            Contains reference names (key) and paths (value) to
            preprocessed anatomical files.
            Required keys:
            -   [mask-brain] = path to fmriprep brain mask
            -   [mask-probCS] = path to fmriprep CSF label
            -   [mask-probWM] = path to fmriprep WM label
        func_dict : dict
            Contains reference names (key) and paths (value) to
            preprocessed functional files.
            Required keys:
            -   [func-preproc] = list of fmriprep preprocessed EPI paths

        Attributes
        ----------
        _sing_prep : list
            First part of subprocess call for AFNI singularity call
        _task : str
            BIDS task identifier

        Raises
        ------
        KeyError
            Missing expected key in anat_dict or func_dict

        """
        print("\nInitializing MakeMasks")

        # Validate dict keys
        for _key in ["mask-brain", "mask-probCS", "mask-probWM"]:
            if _key not in anat_dict:
                raise KeyError(f"Expected {_key} key in anat_dict")
        if "func-preproc" not in func_dict:
            raise KeyError("Expected func-preproc key in func_dict")

        # Set attributes
        self._subj_work = subj_work
        self._proj_deriv = proj_deriv
        self._anat_dict = anat_dict
        self._func_dict = func_dict
        self._sing_prep = afni_helper.prepend_afni_sing(
            self._proj_deriv, self._subj_work
        )

        try:
            _file_name = os.path.basename(func_dict["func-preproc"][0])
            subj, sess, task, _, _, _, _, _ = _file_name.split("_")
        except ValueError:
            raise ValueError(
                "BIDS file names required for items in func_dict: "
                + "subject, session, task, run, space, resolution, "
                + "description, and suffix.ext BIDS fields are "
                + "required by afni.MakeMasks. "
                + f"\n\tFound : {_file_name}"
            )
        self._subj = subj
        self._sess = sess
        self._task = task

    def intersect(self, c_frac=0.5, nbr_type="NN2", nbr_num=17):
        """Create an func-anat intersection mask.

        Generate a binary mask for voxels associated with both
        preprocessed anat and func data.

        Parameters
        ----------
        c_fract : float, optional
            Clip level fraction for AFNI's 3dAutomask
        nbr_type : str, optional
            [NN1 | NN2 | NN3]
            Nearest-neighbor type for AFNI's 3dAautomask
        nbr_num : int, optional
            Number of neibhors needed to avoid eroding in
            AFNI's 3dAutomask.

        Raises
        ------
        TypeError
            Invalid types for optional args
        ValueError
            Invalid parameters for optional args

        Returns
        -------
        path
            Location of anat-epi intersection mask

        """
        print("\tMaking intersection mask")

        # Validate arguments
        if not isinstance(c_frac, float):
            raise TypeError("c_frac must be float type")
        if not isinstance(nbr_type, str):
            raise TypeError("nbr_frac must be str type")
        if not isinstance(nbr_num, int):
            raise TypeError("nbr_numc must be int type")
        if c_frac < 0.1 or c_frac > 0.9:
            raise ValueError("c_fract must be between 0.1 and 0.9")
        if nbr_type not in ["NN1", "NN2", "NN3"]:
            raise ValueError("nbr_type must be NN1 | NN2 | NN3")
        if nbr_num < 6 or nbr_num > 26:
            raise ValueError("nbr_num must be between 6 and 26")

        # Setup output path, avoid repeating work
        out_path = (
            f"{self._subj_work}/{self._subj}_"
            + f"{self._sess}_{self._task}_desc-intersect_mask.nii.gz"
        )
        if os.path.exists(out_path):
            return out_path

        # Make binary masks for each preprocessed func file
        auto_list = []
        for run_file in self._func_dict["func-preproc"]:
            h_name = "tmp_autoMask_" + os.path.basename(run_file)
            h_out = os.path.join(self._subj_work, h_name)
            if not os.path.exists(h_out):
                bash_list = [
                    "3dAutomask",
                    f"-clfrac {c_frac}",
                    f"-{nbr_type}",
                    f"-nbhrs {nbr_num}",
                    f"-prefix {h_out}",
                    run_file,
                ]
                bash_cmd = " ".join(self._sing_prep + bash_list)
                _ = submit.submit_subprocess(bash_cmd, h_out, "Automask")
            auto_list.append(h_out)

        # Generate a union mask from the preprocessed masks
        union_out = os.path.join(
            self._subj_work, f"tmp_{self._task}_union.nii.gz"
        )
        if not os.path.exists(union_out):
            bash_list = [
                "3dmask_tool",
                f"-inputs {' '.join(auto_list)}",
                "-union",
                f"-prefix {union_out}",
            ]
            bash_cmd = " ".join(self._sing_prep + bash_list)
            _ = submit.submit_subprocess(bash_cmd, union_out, "Union")

        # Make anat-func intersection mask from the union and
        # fmriprep brain mask.
        bash_list = [
            "3dmask_tool",
            f"-input {union_out} {self._anat_dict['mask-brain']}",
            "-inter",
            f"-prefix {out_path}",
        ]
        bash_cmd = " ".join(self._sing_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_path, "Intersect")
        return out_path

    def tissue(self, thresh=0.5):
        """Make eroded tissue masks.

        Generate eroded white matter and CSF masks.

        Parameters
        ----------
        thresh : float, optional
            Threshold for binarizing probabilistic tissue
            mask output by fMRIPrep.

        Raises
        ------
        TypeError
            Inappropriate types for optional args
        ValueError
            Inappropriate value for optional args

        Returns
        -------
        dict
            ["CS"] = /path/to/eroded/CSF/mask
            ["WM"] = /path/to/eroded/WM/mask

        """
        # Validate args
        if not isinstance(thresh, float):
            raise TypeError("thresh must be float type")
        if thresh < 0.01 or thresh > 0.99:
            raise ValueError("thresh must be between 0.01 and 0.99")

        # Make CSF and WM masks
        out_dict = {"CS": "", "WM": ""}
        for tiss in out_dict.keys():
            print(f"\tMaking eroded tissue mask : {tiss}")

            # Setup final path, avoid repeating work
            out_tiss = os.path.join(
                self._subj_work,
                f"{self._subj}_{self._sess}_label-{tiss}e_mask.nii.gz",
            )
            if os.path.exists(out_tiss):
                out_dict[tiss] = out_tiss
                continue

            # Binarize probabilistic tissue mask
            in_path = self._anat_dict[f"mask-prob{tiss}"]
            bin_path = os.path.join(self._subj_work, f"tmp_{tiss}_bin.nii.gz")
            bash_list = [
                "c3d",
                in_path,
                f"-thresh {thresh} 1 1 0",
                f"-o {bin_path}",
            ]
            bash_cmd = " ".join(bash_list)
            _ = submit.submit_subprocess(
                bash_cmd, bin_path, f"Binarize {tiss}"
            )

            # Eroded tissue mask
            bash_list = [
                "3dmask_tool",
                f"-input {bin_path}",
                "-dilate_input -1",
                f"-prefix {out_tiss}",
            ]
            bash_cmd = " ".join(self._sing_prep + bash_list)
            _ = submit.submit_subprocess(bash_cmd, bin_path, f"Erode {tiss}")

            # Add path to eroded tissue mask
            out_dict[tiss] = out_tiss
        return out_dict

    def minimum(self):
        """Create a minimum-signal mask.

        Generate a mask for voxels in functional space that
        contain a value greater than some minimum threshold
        across all volumes and runs.

        Based around AFNI's 3dTstat -min.

        Returns
        -----
        path
            Location of minimum-value mask

        """
        print("\tMaking minimum value mask")

        # Setup file path, avoid repeating work
        out_path = (
            f"{self._subj_work}/{self._subj}_"
            + f"{self._sess}_{self._task}_desc-minval_mask.nii.gz"
        )
        if os.path.exists(out_path):
            return out_path

        # Make minimum value mask for each run
        min_list = []
        for run_file in self._func_dict["func-preproc"]:
            # Mask epi voxels that have some data
            h_name_bin = "tmp_bin_" + os.path.basename(run_file)
            h_out_bin = os.path.join(self._subj_work, h_name_bin)
            bash_list = [
                "3dcalc",
                "-overwrite",
                f"-a {run_file}",
                "-expr 1",
                f"-prefix {h_out_bin}",
            ]
            bash_cmd = " ".join(self._sing_prep + bash_list)
            _ = submit.submit_subprocess(bash_cmd, h_out_bin, "Binary EPI")

            # Make a mask for >min values
            h_name_min = "tmp_min_" + os.path.basename(run_file)
            h_out_min = os.path.join(self._subj_work, h_name_min)
            bash_list = [
                "3dTstat",
                "-min",
                f"-prefix {h_out_min}",
                h_out_bin,
            ]
            bash_cmd = " ".join(self._sing_prep + bash_list)
            min_list.append(
                submit.submit_subprocess(bash_cmd, h_out_min, "Minimum EPI")
            )

        # Average the minimum masks across runs
        h_name_mean = (
            f"tmp_{self._subj}_{self._sess}_{self._task}"
            + "_desc-mean_mask.nii.gz"
        )
        h_out_mean = os.path.join(self._subj_work, h_name_mean)
        bash_list = [
            "3dMean",
            "-datum short",
            f"-prefix {h_out_mean}",
            f"{' '.join(min_list)}",
        ]
        bash_cmd = " ".join(self._sing_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, h_out_mean, "Mean EPI")

        # Generate mask of non-zero voxels
        bash_list = [
            "3dcalc",
            f"-a {h_out_mean}",
            "-expr 'step(a-0.999)'",
            f"-prefix {out_path}",
        ]
        bash_cmd = " ".join(self._sing_prep + bash_list)
        _ = submit.submit_subprocess(bash_cmd, out_path, "MinVal EPI")
        return out_path


def group_mask(proj_deriv, subj_list, model_name, out_dir):
    """Generate a group intersection mask.

    Make a union mask of all participant intersection masks. Output
    file is written to:
        <out_dir>/group_<model_name>_intersection_mask.nii.gz

    Parameters
    ----------
    proj_deriv : path
        Location of project derivatives directory
    subj_list : list
        Subjects to include in the mask
    model_name : str
        [univ]
        Model identifier of deconvolved file
    out_dir : path
        Desired output location

    Returns
    -------
    path
        Location of group intersection mask

    Raises
    ------
    ValueError
        If not participant intersection masks are encountered

    """
    # Setup output path
    out_path = os.path.join(
        out_dir, f"group_{model_name}_intersection_mask.nii.gz"
    )
    if os.path.exists(out_path):
        return out_path

    # Identify intersection masks
    mask_list = []
    for subj in subj_list:
        for sess in ["ses-day2", "ses-day3"]:
            mask_path = os.path.join(
                proj_deriv,
                "model_afni",
                subj,
                sess,
                "func",
                f"{subj}_{sess}_desc-intersect_mask.nii.gz",
            )
            if os.path.exists(mask_path):
                mask_list.append(mask_path)
    if not mask_list:
        raise ValueError("Failed to find masks in model_afni")

    # Make group intersection mask
    print("Making group intersection mask")
    bash_list = [
        "3dmask_tool",
        "-frac 1",
        f"-prefix {out_path}",
        f"-input {' '.join(mask_list)}",
    ]
    bash_cmd = " ".join(bash_list)
    _ = submit.submit_subprocess(bash_cmd, out_path, "Group Mask")
    return out_path


def tpl_gm(out_dir):
    """Make a gray matter mask from template priors.

    Make a binary gray matter mask from the Harvard-Oxford cortical
    and subcortical structural atlas. For more information on atlas,
    see https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases.

    Writes output to:
        <out_dir>/tpl_GM_mask.nii.gz

    Parameters
    ----------
    out_dir : path
        Location of output directory

    Returns
    -------
    path
        Location of generated gray matter mask

    Raises
    ------
    FileNotFoundError
        Missing template priors

    Notes
    -----
    Requires templateflow configured in environment and the template
    tpl-MNI152NLin6Asym.

    """
    # Avoid repeating work
    out_name = "tpl_GM_mask"
    out_path = os.path.join(out_dir, f"{out_name}.nii.gz")
    if os.path.exists(out_path):
        return out_path

    # Orient to template priors
    try:
        tplflow_dir = os.environ["SINGULARITYENV_TEMPLATEFLOW_HOME"]
    except KeyError:
        raise EnvironmentError(
            "Expected global variable SINGULARITYENV_TEMPLATEFLOW_HOME"
        )
    tpl_dseg = os.path.join(
        tplflow_dir,
        "tpl-MNI152NLin6Asym",
        "tpl-MNI152NLin6Asym_res-02_atlas-HOSPA_desc-th25_dseg.nii.gz",
    )
    if not os.path.exists(tpl_dseg):
        raise FileNotFoundError(
            f"Missing template segmentation profile : {tpl_dseg}"
        )

    # Find WM, CSF, brainstem labels
    c3d_meth = matrix.C3dMethods(out_dir)
    excl_dict = {1: "lwm", 3: "lcsf", 12: "rwm", 14: "rcsf", 8: "bs"}
    excl_list = []
    for ex_num, ex_name in excl_dict.items():
        excl_list.append(
            c3d_meth.thresh(ex_num, ex_num, 1, 0, tpl_dseg, f"tmp_{ex_name}")
        )

    # Remove WM, CSF from mask, binarize GM
    excl_comb = c3d_meth.comb(excl_list, "tmp_excl")
    excl_bin = c3d_meth.thresh(1, 15, 0, 1, excl_comb, "tmp_excl_bin")
    final_mult = c3d_meth.mult(tpl_dseg, excl_bin, "tmp_final")
    out_path = c3d_meth.thresh(1, 30, 1, 0, final_mult, out_name)

    # Clean intermediates
    tmp_list = glob.glob(f"{out_dir}/tmp_*")
    for tmp_path in tmp_list:
        os.remove(tmp_path)
    return out_path
