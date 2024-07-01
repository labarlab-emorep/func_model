from setuptools import setup, find_packages

exec(open("func_model/_version.py").read())

setup(
    name="func_model",
    version=__version__,  # noqa: F821
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "func_model=func_model.entrypoint:main",
            "afni_model=func_model.cli.afni_model:main",
            "afni_extract=func_model.cli.afni_extract:main",
            "afni_etac=func_model.cli.afni_etac:main",
            "afni_mvm=func_model.cli.afni_mvm:main",
            "fsl_model=func_model.cli.fsl_model:main",
            "fsl_extract=func_model.cli.fsl_extract:main",
            "fsl_map=func_model.cli.fsl_map:main",
            "fsl_group=func_model.cli.fsl_group:main",
        ]
    },
    include_package_data=True,
    package_data={"": ["reference_files/design*.fsf"]},
    install_requires=[
        "natsort>=8.4.0",
        "nibabel>=4.0.1",
        "numpy>=1.22.3",
        "pandas>=1.4.4",
        "setuptools>=65.5.0",
    ],
)
