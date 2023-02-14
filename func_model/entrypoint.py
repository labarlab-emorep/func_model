"""Print entrypoint help."""


def main():
    print(
        """

    The package func_model consists of sub-packages that can be accessed
    from their respective entrypoints:

        afni_model   : Conduct AFNI-style deconvolution
        afni_extract : Extract emotion beta values from deconvolve AFNI as a matrix
        afni_univ    : Conduct univariate tests in AFNI
        afni_mvm     : Conduct multivariate tests in AFNI

    Sub-packages under development:

        fsl_model    : Conduct FSL-style first- and second-level regressions
        fsl_extract  : Extract emotion beta values from FSL first-level as a matrix
        
    The sub-packages afni_model, fsl_model are written specifically for the
    Duke Compute Cluster (DCC), while afni_extract, afni_univ, afni_mvm, and
    fsl_extract are written for labarserv2.
    
    
    """
    )


if __name__ == "__main__":
    main()
