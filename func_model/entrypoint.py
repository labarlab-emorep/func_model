"""Print entrypoint help."""
import func_model._version as ver


def main():
    print(
        f"""

    Version : {ver.__version__}

    The package func_model consists of sub-packages that can be accessed
    from their respective entrypoints:

        afni_model   : Conduct AFNI-style deconvolution
        afni_extract : Extract emotion beta values from deconvolve AFNI as a matrix
        afni_univ    : Conduct univariate tests in AFNI
        afni_mvm     : Conduct multivariate tests in AFNI
        fsl_model    : Conduct FSL-style first-level regressions
        fsl_extract  : Extract emotion beta values from FSL first-level as a matrix
        fsl_map      : Make binary masks from classifier output

    Sub-packages under development:

        fsl_model    : Plan to add second-level and resting-state models

    The sub-packages afni_model, fsl_model are written specifically for the
    Duke Compute Cluster (DCC), while afni_extract, afni_univ, afni_mvm,
    fsl_extract, and fsl_map are written for labarserv2.

    """
    )


if __name__ == "__main__":
    main()
