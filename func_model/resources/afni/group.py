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
    get_labels(emo_name=None)
        Derive relevant sub-brick identifiers
    split_decon(emo_name=None)
        Split deconvolve file into each relevant sub-brick
    mask_coord()
        Find coordinates to censor based on brain mask
    make_func_matrix(subj, sess, task, model_name, decon_path)
        Generate dataframe of voxel beta weights for subject, session

    Example
    -------
    etb_obj = group.ExtractTaskBetas(*args)
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

    def get_label_names(self):
        """Get sub-brick levels from AFNI deconvolved file.

        Attributes
        ----------
        stim_label : list
            Full label ID of sub-brick starting with "mov" or "sce",
            e.g. movFea#0_Coef.

        Raises
        ------
        ValueError
            Trouble parsing output of 3dinfo -label to list

        Example
        -------
        etb_obj = group.ExtractTaskBetas(*args)
        etb_obj.decon_path = "/path/to/decon/file+tlrc"
        etb_obj.subj_out_dir = "/path/to/output/dir"
        etb_obj.get_label_names()

        """
        # Extract sub-brick label info
        out_label = os.path.join(self.subj_out_dir, "tmp_labels.txt")
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
        os.remove(out_label)

        # Identify labels relevant to task
        self.stim_label = [
            x
            for x in label_list
            if self.task.split("-")[1][:3] in x and "Fstat" not in x
        ]
        self.stim_label.sort()

    def get_label_int(self, sub_label):
        """Return sub-brick ID of requested sub-brick label.

        Parameters
        ----------
        sub_label : str
            Sub-brick label, e.g. movFea#0_Coef

        Returns
        -------
        int

        Raises
        ------
        ValueError
            Trouble in parsing 3dinfo output

        Example
        -------
        etb_obj = group.ExtractTaskBetas(*args)
        etb_obj.decon_path = "/path/to/decon/file"
        etb_obj.subj_out_dir = "/path/to/output/dir"
        sub_int = etb_obj.get_label_int("movFea#0_Coef")

        """
        # Determine sub-brick integer value
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

        # Read label integer and check
        with open(out_label, "r") as lf:
            label_int = lf.read().strip()
        if len(label_int) > 2:
            raise ValueError(f"Unexpected int length for {sub_label}")
        os.remove(out_label)
        return label_int

    def split_decon(self, emo_name=None):
        """Split deconvolved files into files by sub-brick.

        Parameters
        ----------
        emo_name: str, optional
            Long name of emotion, for restricting beta extraction
            to single emotion

        Attributes
        ----------
        beta_dict : dict
            key = emotion, value = path to sub-brick file

        Raises
        ------
        ValueError
            Sub-brick identifier int has length > 2

        Example
        -------
        etb_obj = group.ExtractTaskBetas(**args)
        etb_obj.subj = "sub-100"
        etb_obj.sess = "ses-1"
        etb_obj.task = "task-movies"
        etb_obj.decon_path = "/path/to/decon/file+tlrc"
        etb_obj.subj_out_dir = "/path/to/output/dir"
        etb_obj.split_decon(emo_name="fear")

        """
        # Invert emo_switch to unpack sub-bricks
        _emo_switch = helper.emo_switch()
        emo_switch = {j: i for i, j in _emo_switch.items()}

        # Extract desired sub-bricks from deconvolve file by label name
        self.get_label_names()
        self.beta_dict = {}
        for sub_label in self.stim_label:

            # Identify emo string
            emo_long = emo_switch[sub_label[3:6]]
            if emo_name and emo_long != emo_name:
                continue

            # Determine sub-brick integer value
            label_int = self.get_label_int(sub_label)

            # Setup file name, avoid repeating beta extraction
            out_file = (
                f"tmp_{self.subj}_{self.sess}_{self.task}_"
                + f"desc-{emo_long}_beta.nii.gz"
            )
            out_path = os.path.join(self.subj_out_dir, out_file)
            if os.path.exists(out_path):
                self.beta_dict[emo_long] = out_path
                continue

            # Write sub-brick as new file
            print(f"\t\tExtracting sub-brick for {emo_long}")
            bash_list = [
                "3dTcat",
                f"-prefix {out_path}",
                f"{self.decon_path}[{label_int}]",
                "> /dev/null 2>&1",
            ]
            bash_cmd = " ".join(bash_list)
            _ = submit.submit_subprocess(bash_cmd, out_path, "Split decon")
            self.beta_dict[emo_long] = out_path

    # def mask_coord(self, mask_path):
    #     """Identify censoring coordinates from binary brain mask.

    #     Read-in binary values from a brain mask, vectorize, and identify
    #     coordinates of mask file outside of brain. Sets internal attribute
    #     holding coordinates to remove from beta dataframes.

    #     Parameters
    #     ----------
    #     mask_path : path
    #         Location of binary brain mask

    #     Attributes
    #     ----------
    #     _rm_cols : array
    #         Column names (coordinates) to drop from beta dataframes

    #     """
    #     print("\tFinding coordinates to censor ...")
    #     img_flat = self.nifti_to_arr(mask_path)
    #     df_mask = self.arr_to_df(img_flat)
    #     self._rm_cols = df_mask.columns[df_mask.isin([0.0]).any()]

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
        if not helper.valid_task(task):
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
        for emo, beta_path in self.beta_dict.items():
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


class EtacTest:
    """Build and execute ETAC tests.

    Identify relevant sub-bricks in AFNI's deconvolved files given user
    input, then construct and run ETAC tests. ETAC shell script and
    output files are written to <out_dir>.

    Methods
    --------
    write_exec(*args)
        Construct and run a T-test via ETAC (3dttest++)

    Example
    -------
    et_obj = group.EtacTest(*args)
    _ = et_obj.write_exec(*args)

    """

    def __init__(self, proj_dir, out_dir, mask_path):
        """Initialize.

        Parameters
        ----------
        proj_dir : path
            Location of project directory
        out_dir : path
            Output location for generated files
        mask_path : path
            Location of group mask

        Raises
        ------
        FileNotFoundError
            Missing proj_dir

        """
        print("Initializing EtacTest")

        # Check and setup
        if not os.path.exists(proj_dir):
            raise FileNotFoundError(f"Missing expected proj_dir : {proj_dir}")
        self._proj_dir = proj_dir
        self._out_dir = out_dir
        self._mask_path = mask_path

    def _build_list(self, decon_dict: dict, sub_label: str) -> list:
        """Build ETAC input list."""
        set_list = []
        get_subbrick = ExtractTaskBetas(self._proj_dir)
        for subj, decon_path in decon_dict.items():

            # Get label integer
            get_subbrick.decon_path = decon_path
            get_subbrick.subj_out_dir = self._out_dir
            label_int = get_subbrick.get_label_int(sub_label)

            # Write input line
            set_list.append(subj)
            set_list.append(f"{decon_path}'[{label_int}]'")
        return set_list

    def _validate_etac_input(self, group_dict: dict, sub_label: str):
        """Check that user input conforms."""
        # Check group dict structure
        for _subj, info_dict in group_dict.items():
            if "decon_path" not in info_dict.keys():
                raise KeyError("Improper structure of group_dict")

        # Check specified sub_label
        _nam, chk_str = sub_label.split("#")
        if chk_str != "0_Coef":
            raise ValueError("Improper format of sub_label")

        # Check model name
        if not helper.valid_univ_test(self._model_name):
            raise ValueError("Improper model name specified")

    def _etac_opts(
        self,
        final_name: str,
        blur: int = 0,
    ) -> list:
        """Return ETAC options."""
        etac_head = [f"cd {self._out_dir};", "3dttest++"]
        etac_body = [
            f"-mask {self._mask_path}",
            f"-prefix {final_name}",
            f"-prefix_clustsim {final_name}_clustsim",
            "-ETAC",
            f"-ETAC_blur {blur}",
            "-ETAC_opt",
            "NN=2:sid=2:hpow=0:pthr=0.01,0.005,0.002,0.001:name=etac",
        ]
        if self._model_name == "paired":
            return etac_head + ["-paired"] + etac_body
        else:
            return etac_head + etac_body

    def _write_script(self, final_name: str, bash_cmd: str):
        """Write BASH command to shell script."""
        etac_script = os.path.join(self._out_dir, f"{final_name}.sh")
        with open(etac_script, "w") as script:
            script.write(bash_cmd)

    def _setup(self, emo_short: str) -> tuple:
        """Return final filename and location."""
        print(f"\tBuilding {self._model_name} ETAC for {emo_short}")
        if not os.path.exists(self._out_dir):
            os.makedirs(self._out_dir)
        final_name = f"FINAL_{self._model_name}_{emo_short}"
        out_path = os.path.join(self._out_dir, f"{final_name}+tlrc.HEAD")
        return (final_name, out_path)

    def write_exec(self, model_name, emo_short, group_dict, sub_label):
        """Write and execute a T-test using AFNI's ETAC method.

        Compare coefficient (sub_label) against null for model_name=student
        or against the washout coefficient for model_name=paired. Generates
        the ETAC command (3dttest++), writes it to a shell script for review,
        and then executes the command.

        Output files are saved to:
            <out_dir>/FINAL_<model_name>_<emo_short>

        Parameters
        ----------
        model_name : str
            [paired | student]
            Model identifier
        emo_short : str
            Shortened (AFNI) emotion name
        group_dict : dict
            Group information in format:
            {"sub-ER0009": {"decon_path": "/path/to/decon+tlrc.HEAD"}}
        sub_label : str
            Sub-brick label, e.g. movFea#0_Coef

        Returns
        -------
        path
            Location of output directory

        """
        # Validate and setup
        self._model_name = model_name
        self._validate_etac_input(group_dict, sub_label)
        final_name, out_path = self._setup(emo_short)
        if os.path.exists(out_path):
            return self._out_dir

        # Make input list
        decon_dict = {}
        for subj, decon_info in group_dict.items():
            decon_dict[subj] = decon_info["decon_path"]
        setA_list = self._build_list(decon_dict, sub_label)
        etac_set = [f"-setA {emo_short}", " ".join(setA_list)]
        if model_name == "paired":
            setB_list = self._build_list(decon_dict, "comWas#0_Coef")
            etac_set = etac_set + ["-setB washout", " ".join(setB_list)]

        # Build ETAC command, write to script for review
        etac_list = self._etac_opts(final_name) + etac_set
        etac_cmd = " ".join(etac_list)
        self._write_script(final_name, etac_cmd)

        # Execute ETAC command
        submit.submit_subprocess(etac_cmd, out_path, f"etac{emo_short}")
        return self._out_dir
