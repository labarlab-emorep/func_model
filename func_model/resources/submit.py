"""Methods for controlling sbatch and subprocess submissions.

submit_subprocess : submit and check for output of bash command
submit_sbatch : schedule bash command with Slurm
schedule_afni : schedule AFNI workflow with Slurm
schedule_fsl : schedule FSL workflow with Slurm
schedule_afni_group_setup : download data from Keoki
schedule_afni_group_subbrick : mine deconvolve files for subbrick IDs
schedule_afni_group_etac : conduct t-testing via ETAC (3dttest++)
schedule_afni_group_lmer : conduct linear mixed effects models via 3dLMEr

"""

import os
import sys
import subprocess
import textwrap
from typing import Union
from func_model.resources import helper

# TODO refactor schedule_afni_group_* methods into class


def submit_subprocess(bash_cmd, chk_path, job_name, force_cont=False):
    """Submit bash command as subprocess.

    Check for output file after submission, print stdout
    and stderr if output file is not found.

    Parameters
    ----------
    bash_cmd : str
        Bash command
    chk_path : path
        Location of generated file
    job_name : str
        Identifier for error messages
    force_cont : bool, optional
        Skip file check, raising FileNotFoundError

    Returns
    -------
    str, os.PathLike, None
        Location of generated file

    Raises
    ------
    FileNotFoundError
        Expected output file not detected

    """
    h_sp = subprocess.Popen(
        bash_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    h_out, h_err = h_sp.communicate()
    h_sp.wait()
    if force_cont:
        return
    if not os.path.exists(chk_path):
        print(
            f"""\n
        {job_name} failed!
            command : {bash_cmd}
            stdout : {h_out}
            stderr : {h_err}\n
        """
        )
        raise FileNotFoundError(f"Expected to find file : {chk_path}")
    return chk_path


def submit_sbatch(
    bash_cmd,
    job_name,
    log_dir,
    num_hours=1,
    num_cpus=1,
    mem_gig=4,
    env_input=None,
):
    """Schedule child SBATCH job.

    Parameters
    ----------
    bash_cmd : str
        Bash syntax, work to schedule
    job_name : str
        Name for scheduler
    log_dir : Path
        Location of output dir for writing logs
    num_hours : int
        Walltime to schedule
    num_cpus : int
        Number of CPUs required by job
    mem_gig : int
        Job RAM requirement (GB)
    env_input : dict, None
        Extra environmental variables required by processes
        e.g. singularity reqs

    Returns
    -------
    tuple
        [0] = stdout of subprocess
        [1] = stderr of subprocess

    Notes
    -----
    Avoid using double quotes in <bash_cmd> (particularly relevant
    with AFNI) to avoid conflict with --wrap syntax.

    """
    sbatch_cmd = f"""
        sbatch \
        -J {job_name} \
        -t {num_hours}:00:00 \
        --cpus-per-task={num_cpus} \
        --mem={mem_gig}G \
        -o {log_dir}/out_{job_name}.log \
        -e {log_dir}/err_{job_name}.log \
        --wait \
        --wrap="{bash_cmd}"
    """
    print(f"Submitting SBATCH job:\n\t{sbatch_cmd}\n")
    h_sp = subprocess.Popen(
        sbatch_cmd, shell=True, stdout=subprocess.PIPE, env=env_input
    )
    h_out, h_err = h_sp.communicate()
    h_sp.wait()
    return (h_out, h_err)


def schedule_afni(
    subj,
    sess,
    work_deriv,
    model_name,
    log_dir,
):
    """Write and schedule pipeline.

    Generate a python script that controls preprocessing. Submit
    the work on schedule resources. Writes parent script to:
        log_dir/run_model-afni_subj_sess.py

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    work_deriv : path
        Parent location for writing pipeline intermediates
    model_name : str
        {"rest", "mixed", "task", "block"}
        Desired AFNI model, for triggering different workflows
    log_dir : path
        Output location for log files and scripts

    Returns
    -------
    tuple
        [0] subprocess stdout
        [1] subprocess stderr

    """
    # Setup software derivatives dirs, for working
    work_afni = os.path.join(work_deriv, "model_afni")
    if not os.path.exists(work_afni):
        os.makedirs(work_afni)

    # Write parent python script
    pipe_name = "rest" if model_name == "rest" else "task"
    sbatch_cmd = f"""\
        #!/bin/env {sys.executable}

        #SBATCH --job-name=p{subj[6:]}s{sess[-1]}
        #SBATCH --output={log_dir}/par{subj[6:]}s{sess[-1]}.txt
        #SBATCH --time=45:00:00
        #SBATCH --mem=12G

        import os
        import sys
        from func_model.workflows import wf_afni

        wf_afni.afni_{pipe_name}(
            "{subj}",
            "{sess}",
            "{work_deriv}",
            "{model_name}",
            "{log_dir}",
        )

    """
    sbatch_cmd = textwrap.dedent(sbatch_cmd)
    py_script = f"{log_dir}/run-afni_model-{model_name}_{subj}_{sess}.py"
    with open(py_script, "w") as ps:
        ps.write(sbatch_cmd)

    # Execute script
    h_sp = subprocess.Popen(
        f"sbatch {py_script}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    h_out, h_err = h_sp.communicate()
    print(f"{h_out.decode('utf-8')}\tfor {subj} {sess}")
    return (h_out, h_err)


def schedule_fsl(
    subj,
    sess,
    model_name,
    model_level,
    preproc_type,
    proj_rawdata,
    proj_deriv,
    work_deriv,
    log_dir,
):
    """Write and schedule pipeline.

    Generate a python script that controls FSL FEAT models. Submit
    the work on schedule resources. Writes parent script to:
        log_dir/run-fsl_model-<model_name>_level-<model_level>_subj_sess.py

    Parameters
    ----------
    subj : str
        BIDS subject identifier
    sess : str
        BIDS session identifier
    model_name : str
        Name of FSL model, for keeping condition files and
        output organized
    model_level : str
        Level of FSL model
    preproc_type : str
        [smoothed | scaled]
        Select preprocessed EPI file to model
    proj_rawdata : path
        Location of BIDS rawdata
    proj_deriv : path
        Location of project BIDs derivatives, for finding
        preprocessed output
    work_deriv : path
        Output location for intermediates
    log_dir : path
        Output location for log files and scripts

    Returns
    -------
    tuple
        [0] subprocess stdout
        [1] subprocess stderr

    Raises
    ------
    ValueError
        Unexpected argument parameters

    """
    if not helper.valid_name(model_name):
        raise ValueError(f"Unexpected model name : {model_name}")
    if not helper.valid_level(model_level):
        raise ValueError(f"Unexpected model level : {model_level}")

    def _sbatch_head() -> str:
        """Return script header."""
        subj_short = subj[6:]
        sess_short = sess[-1]
        return f"""\
            #!/bin/env {sys.executable}
            #SBATCH --job-name=p{subj_short}s{sess_short}
            #SBATCH --output={log_dir}/par{subj_short}s{sess_short}.txt
            #SBATCH --time=30:00:00
            #SBATCH --mem=8G

            from func_model.workflows import wf_fsl
        """

    def _first_call() -> str:
        """Return first-level wf call."""
        wf_meth = "model_rest" if model_name == "rest" else "model_task"
        return f"""{_sbatch_head()}
            wf_obj = wf_fsl.FslFirst(
                "{subj}",
                "{sess}",
                "{model_name}",
                "{preproc_type}",
                "{proj_rawdata}",
                "{proj_deriv}",
                "{work_deriv}",
                "{log_dir}",
            )
            wf_obj.{wf_meth}()
        """

    def _second_call() -> str:
        """Return second-level wf call."""
        return f"""{_sbatch_head()}
            wf_obj = wf_fsl.FslSecond(
                "{subj}",
                "{sess}",
                "{model_name}",
                "{proj_deriv}",
                "{work_deriv}",
                "{log_dir}",
            )
            wf_obj.model_task()
        """

    # Trigger appropriate inner function for model_level
    sbatch_cmd = (
        textwrap.dedent(_first_call())
        if model_level == "first"
        else textwrap.dedent(_second_call())
    )
    py_script = (
        f"{log_dir}/run-fsl_model-{model_name}_"
        + f"level-{model_level}_{subj}_{sess}.py"
    )
    with open(py_script, "w") as ps:
        ps.write(sbatch_cmd)

    # Execute script
    h_sp = subprocess.Popen(
        f"sbatch {py_script}",
        shell=True,
        stdout=subprocess.PIPE,
    )
    h_out, h_err = h_sp.communicate()
    print(f"{h_out.decode('utf-8')}\tfor {subj}, {sess}")
    return (h_out, h_err)


def schedule_afni_group_setup(
    task: str,
    model_name: str,
    work_deriv: Union[str, os.PathLike],
    log_dir: Union[str, os.PathLike],
):
    """Coordinate setup for group-level AFNI modelling."""
    # Write parent python script
    sbatch_cmd = f"""\
        #!/bin/env {sys.executable}

        #SBATCH --job-name=pAfniSetup
        #SBATCH --output={log_dir}/parAfniSetup.txt
        #SBATCH --time=10:00:00
        #SBATCH --mem=8G

        from func_model.resources import helper
        from func_model.resources import masks

        # Get modeled data
        get_data = helper.SyncGroup("{work_deriv}")
        _, model_group = get_data.setup_group()
        get_data.get_model_afni("{task}", "{model_name}")

        # Make template GM mask
        masks.tpl_gm(model_group)

    """
    sbatch_cmd = textwrap.dedent(sbatch_cmd)
    py_script = f"{log_dir}/run-afni_group-setup.py"
    with open(py_script, "w") as ps:
        ps.write(sbatch_cmd)

    # Execute script
    h_sp = subprocess.Popen(
        f"sbatch {py_script}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    h_out, _ = h_sp.communicate()
    print(h_out.decode("utf-8"))


def schedule_afni_group_subbrick(
    task: str,
    model_name: str,
    stat: str,
    emo_list: list,
    work_deriv: Union[str, os.PathLike],
    log_dir: Union[str, os.PathLike],
    blk_coef: bool,
):
    """Coordinate setup for group-level AFNI modelling."""
    # Write parent python script
    get_wash = True if stat == "paired" else False
    sbatch_cmd = f"""\
        #!/bin/env {sys.executable}

        #SBATCH --job-name=pAfniSubbrick
        #SBATCH --output={log_dir}/parAfniSubbrick.txt
        #SBATCH --time=10:00:00
        #SBATCH --mem=8G

        from func_model.workflows import wf_afni
        from func_model.resources import helper

        # Setup working dirs
        sync_data = helper.SyncGroup("{work_deriv}")
        model_indiv, _ = sync_data.setup_group()

        # Identify decon subbricks
        dcn_sub = wf_afni.DeconSubbrick(
            "{model_name}",
            model_indiv,
        )
        dcn_sub.find_subbricks(
            "{task}",
            {emo_list},
            {blk_coef},
            {get_wash},
        )

    """
    sbatch_cmd = textwrap.dedent(sbatch_cmd)
    py_script = f"{log_dir}/run-afni_group-subbrick.py"
    with open(py_script, "w") as ps:
        ps.write(sbatch_cmd)

    # Execute script
    h_sp = subprocess.Popen(
        f"sbatch {py_script}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    h_out, _ = h_sp.communicate()
    print(h_out.decode("utf-8"))


def schedule_afni_group_etac(
    task: str,
    model_name: str,
    stat: str,
    emo_name: str,
    work_deriv: Union[str, os.PathLike],
    log_dir: Union[str, os.PathLike],
    blk_coef: bool,
):
    """Schedule ETAC workflow."""
    # Write parent python script
    sbatch_cmd = f"""\
        #!/bin/env {sys.executable}

        #SBATCH --job-name=pUniv{emo_name}
        #SBATCH --output={log_dir}/parUniv{emo_name}.txt
        #SBATCH --time=80:00:00
        #SBATCH --mem=4G

        from func_model.workflows import wf_afni

        # Conduct group-level test
        wf_afni.afni_ttest(
            "{task}",
            "{model_name}",
            "{stat}",
            "{emo_name}",
            {blk_coef},
            "{work_deriv}",
            "{log_dir}",
        )

    """
    sbatch_cmd = textwrap.dedent(sbatch_cmd)
    suff = "_block.py" if blk_coef else ".py"
    py_script = (
        f"{log_dir}/run-afni_{stat}-{model_name}_{task}_emo-{emo_name}{suff}"
    )
    with open(py_script, "w") as ps:
        ps.write(sbatch_cmd)

    # Execute script
    h_sp = subprocess.Popen(
        f"sbatch {py_script}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    h_out, _ = h_sp.communicate()

    # Communicate to user
    job_num = h_out.decode("utf-8")
    print(f"{job_num}\tfor emo : {emo_name}")


def schedule_afni_group_lmer(
    model_name: str,
    emo_list: list,
    work_deriv: Union[str, os.PathLike],
    log_dir: Union[str, os.PathLike],
    blk_coef: bool,
):
    """Schedule 3dMVM."""
    # Write parent python script
    sbatch_cmd = f"""\
        #!/bin/env {sys.executable}

        #SBATCH --job-name=pLMEr
        #SBATCH --output={log_dir}/parLMEr.txt
        #SBATCH --time=80:00:00
        #SBATCH --mem=4G

        from func_model.workflows import wf_afni

        # Conduct group-level test
        wf_afni.afni_lmer(
            "{model_name}",
            {emo_list},
            {blk_coef},
            "{work_deriv}",
            "{log_dir}",
        )

    """
    sbatch_cmd = textwrap.dedent(sbatch_cmd)
    suff = "_block.py" if blk_coef else ".py"
    py_script = f"{log_dir}/run-afni_lmer-{model_name}{suff}"
    with open(py_script, "w") as ps:
        ps.write(sbatch_cmd)

    # Execute script
    h_sp = subprocess.Popen(
        f"sbatch {py_script}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    h_out, _ = h_sp.communicate()

    # Communicate to user
    print(h_out.decode("utf-8"))


def schedule_afni_group_mc(
    model_name: str,
    work_deriv: Union[str, os.PathLike],
    log_dir: Union[str, os.PathLike],
):
    """Schedule monte carlo."""
    # Write parent python script
    sbatch_cmd = f"""\
        #!/bin/env {sys.executable}

        #SBATCH --job-name=pMC
        #SBATCH --output={log_dir}/parMC.txt
        #SBATCH --time=80:00:00
        #SBATCH --mem=12G
        #SBATCH -c 10

        from func_model.workflows import wf_afni

        # Conduct monte carlo sims
        wf_afni.afni_montecarlo(
            "{model_name}",
            "{work_deriv}",
            "{log_dir}",
        )

    """
    sbatch_cmd = textwrap.dedent(sbatch_cmd)
    py_script = f"{log_dir}/run-afni_mc-{model_name}.py"
    with open(py_script, "w") as ps:
        ps.write(sbatch_cmd)

    # Execute script
    h_sp = subprocess.Popen(
        f"sbatch {py_script}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    h_out, _ = h_sp.communicate()

    # Communicate to user
    print(h_out.decode("utf-8"))
