"""Print entrypoint help."""


def main():
    print(
        """

    The package func_model consists of two sub-packages that
    can be accessed from their respective entrypoints (below).

        model_afni : Conduct AFNI-style deconvolution (in development)
        model_fsl  : Conduct FSL-style first- and second-level regressions (in development)

    """
    )


if __name__ == "__main__":
    main()
