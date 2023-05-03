"""Methods for controlling sbatch and subprocess submissions."""
import os
import sys
import subprocess
import textwrap
from func_model.resources import fsl


def submit_subprocess(bash_cmd, chk_path, job_name):
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

    Returns
    -------
    path
        Location of generated file

    Raises
    ------
    FileNotFoundError
        Expected output file not detected

    """
    h_sp = subprocess.Popen(bash_cmd, shell=True, stdout=subprocess.PIPE)
    h_out, h_err = h_sp.communicate()
    h_sp.wait()
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
        Job RAM requirement for each CPU (GB)
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
        --mem-per-cpu={mem_gig}000 \
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
    proj_rawdata,
    proj_deriv,
    work_deriv,
    sing_afni,
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
    proj_rawdata : path
        Location of BIDS-organized project rawdata
    proj_deriv : path
        Location of project derivatives, containing fmriprep
        and fsl_denoise sub-directories
    work_deriv : path
        Parent location for writing pipeline intermediates
    sing_afni : path
        Location of AFNI singularity file
    model_name : str
        [univ | rest | mixed]
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
    pipe_name = "rest" if model_name == "rest" else "task"
    work_afni = os.path.join(work_deriv, f"model_afni-{pipe_name}")
    if not os.path.exists(work_afni):
        os.makedirs(work_afni)

    # Setup software derivatives dirs, for storage
    proj_afni = os.path.join(proj_deriv, "model_afni")
    if not os.path.exists(proj_afni):
        os.makedirs(proj_afni)

    # Write parent python script
    wall_time = 20
    sbatch_cmd = f"""\
        #!/bin/env {sys.executable}

        #SBATCH --job-name=p{subj[6:]}s{sess[-1]}
        #SBATCH --output={log_dir}/par{subj[6:]}s{sess[-1]}.txt
        #SBATCH --time={wall_time}:00:00
        #SBATCH --mem=8000

        import os
        import sys
        from func_model import workflows

        _, _, _ = workflows.afni_{pipe_name}(
            "{subj}",
            "{sess}",
            "{proj_rawdata}",
            "{proj_deriv}",
            "{work_deriv}",
            "{sing_afni}",
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
    user_name,
    rsa_key,
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

    TODO
    preproc_type : str
        [smoothed | scaled]

    proj_rawdata : path
        Location of BIDS rawdata
    proj_deriv : path
        Location of project BIDs derivatives, for finding
        preprocessed output
    work_deriv : path
        Output location for intermediates
    log_dir : path
        Output location for log files and scripts

    TODO
    user_name
    rsa_key

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
    if not fsl.helper.valid_name(model_name):
        raise ValueError(f"Unexpected model name : {model_name}")
    if not fsl.helper.valid_level(model_level):
        raise ValueError(f"Unexpected model level : {model_level}")

    # Determine workflow methods
    wf_class = f"Fsl{model_level.capitalize()}"
    wf_meth = "model_rest" if model_name == "rest" else "model_task"

    # Write parent python script
    subj_short = subj[6:]
    sess_short = sess[-1]
    sbatch_cmd = f"""\
        #!/bin/env {sys.executable}

        #SBATCH --job-name=p{subj_short}s{sess_short}
        #SBATCH --output={log_dir}/par{subj_short}s{sess_short}.txt
        #SBATCH --time=30:00:00
        #SBATCH --mem=8000

        import os
        import sys
        from func_model import workflows

        wf_obj = workflows.{wf_class}(
            "{subj}",
            "{sess}",
            "{model_name}",
            "{model_level}",
            "{preproc_type}",
            "{proj_rawdata}",
            "{proj_deriv}",
            "{work_deriv}",
            "{log_dir}",
            "{user_name}",
            "{rsa_key}",
        )
        wf_obj.{wf_meth}()

    """
    sbatch_cmd = textwrap.dedent(sbatch_cmd)
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
