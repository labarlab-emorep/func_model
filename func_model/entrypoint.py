"""Print entrypoint help."""


def main():
    print(
        """

    The package func_model consists of two sub-packages that
    can be accessed from their respective entrypoints (below).
    All sub-packages are under development.

        afni_model   : Conduct AFNI-style deconvolution
        afni_extract : Extract emotion beta values from deconvolve AFNI as a matrix
        afni_univ    : Conduct univariate tests in AFNI
        fsl_model    : Conduct FSL-style first- and second-level regressions
        fsl_extract  : Extract emotion beta values from FSL first-level as a matrix

    """
    )


if __name__ == "__main__":
    main()
