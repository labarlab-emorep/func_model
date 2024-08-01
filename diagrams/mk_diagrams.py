# %%
from diagrams import Cluster, Diagram, Edge

# %%
from diagrams.aws.analytics import DataPipeline
from diagrams.aws.compute import Batch, Compute
from diagrams.aws.database import Database
from diagrams.aws.devtools import CommandLineInterface
from diagrams.aws.general import General

# %%
with Diagram("imports", direction="TB", show=False):

    with Cluster("cli"):
        cli_model = General("fsl_model")
        cli_extract = General("fsl_extract")
        cli_map = General("fsl_map")
        cli_group = General("fsl_group")

    with Cluster("workflows"):
        wf = General("wf_fsl")

    with Cluster("resources.afni"):
        afni_help = General("helper")
        afni_masks = General("masks")

    with Cluster("resources.fsl"):
        helper = General("helper")
        model = General("model")
        group = General("group")

    with Cluster("resources.general"):
        sub = General("submit")
        mat = General("matrix")
        sql = General("sql_database")

    ref_files = General("reference_files")

    # fsl_model
    cli_model << sub << wf
    cli_model << helper
    sub << helper << ref_files

    wf << afni_masks
    afni_masks << afni_help
    afni_masks << sub
    afni_masks << mat << sub

    wf << model
    model << sub
    model << helper

    wf << group
    group << helper
    group << mat
    group << sub
    group << sql

    # other clis
    cli_extract << wf
    cli_map << helper
    cli_map << wf
    cli_group << wf

    # %%
    # graph_attr = {
    #     "layout": "dot",
    #     "compound": "true",
    # }

    # with Diagram("process", graph_attr=graph_attr, show=True):
    #     cli = CommandLineInterface("cli")
    #     with Cluster("submit"):
    #         sb = Batch("schedule_subj")
    #     with Cluster("sbatch_parent"):
    #         wf = Compute("workflows")
    #         with Cluster("run_preproc"):
    #             dl = Database("pull_rawdata")
    #             with Cluster("preprocess"):
    #                 fs = Batch("RunFreeSurfer")
    #                 fp = Batch("RunFmriprep")
    #                 fl = Compute("fsl_preproc")
    #                 with Cluster("helper_tools"):
    #                     ht = Batch("AfniFslMethods")
    #             ul = Database("push_derivatives")
    #     with Cluster("sbatch_child"):
    #         fschild = Compute("recon_all")
    #         fpchild = Compute("fmriprep")
    #         fslchild = DataPipeline("tmean-->scale")

    #     cli >> sb >> Edge(lhead="cluster_sbatch_parent") >> wf
    #     wf >> Edge(lhead="cluster_run_preproc") >> dl
    #     dl >> Edge(lhead="cluster_preprocess") >> fs
    #     fl >> Edge(lhead="cluster_helper_tools") >> ht
    #     fs >> fp >> fl
    #     ht >> Edge(ltail="cluster_preprocess") >> ul

    #     #
    #     fs >> fschild
    #     fp >> fpchild
    # ht >> fslchild


# %%
