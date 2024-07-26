"""Print entrypoint help."""

import func_model._version as ver


def main():
    print(
        f"""

    Version : {ver.__version__}

    The package func_model consists of sub-packages that can be accessed
    from their respective entrypoints:

        afni_model   : Conduct AFNI-style deconvolution
        afni_etac    : Conduct T-tests in AFNI via ETAC
        afni_lmer    : Conduct linear mixed effects in AFNI

        fsl_model    : Conduct FSL-style first- and second-level regressions
        fsl_extract  : Extract emotion betas from FSL first-level as a matrix
        fsl_map      : Make binary masks from classifier output
        fsl_group    : Generate required input for group-level analyses

    Sub-packages written for Duke Compute Cluster (DCC):

        - afni_model
        - afni_etac
        - afni_lmer
        - fsl_model

    Sub-packages written for labarserv2:

        - fsl_extract
        - fsl_map
        - fsl_group

    """
    )


if __name__ == "__main__":
    main()
