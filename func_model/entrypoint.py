"""Print entrypoint help."""

import func_model._version as ver


def main():
    print(
        f"""

    Version : {ver.__version__}

    The package func_model consists of sub-packages that can be accessed
    from their respective entrypoints:

        afni_model   : Conduct AFNI-style deconvolution
        afni_extract : Extract emotion betas from deconvolve AFNI as a matrix
        afni_univ    : Conduct univariate tests in AFNI
        afni_mvm     : Conduct multivariate tests in AFNI

        fsl_model    : Conduct FSL-style first- and second-level regressions
        fsl_extract  : Extract emotion betas from FSL first-level as a matrix
        fsl_map      : Make binary masks from classifier output
        fsl_group    : Generate required input for group-level analyses

    Sub-packages written for Duke Compute Cluster (DCC):

        - afni_model
        - afni_univ
        - fsl_model

    Sub-packages written for labarserv2:

        - afni_extract
        - afni_mvm
        - fsl_extract
        - fsl_map
        - fsl_group

    """
    )


if __name__ == "__main__":
    main()
