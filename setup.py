from setuptools import setup, find_packages

setup(
    name="func_model",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "func_model=func_model.entrypoint:main",
            "afni_model=func_model.cli.afni_model:main",
            "afni_extract=func_model.cli.afni_extract:main",
            "afni_univ=func_model.cli.afni_univ:main",
            "afni_mvm=func_model.cli.afni_mvm:main",
            "fsl_model=func_model.cli.fsl_model:main",
        ]
    },
    include_package_data=True,
    package_data={"": ["reference_files/first_level_design_template.fsf"]},
    install_requires=[
        "nibabel>=4.0.1",
        "numpy>=1.22.3",
        "pandas>=1.4.4",
        "setuptools>=65.5.0",
    ],
)
