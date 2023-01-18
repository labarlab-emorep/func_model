from setuptools import setup, find_packages

setup(
    name="func_model",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "func_model=func_model.entrypoint:main",
            "model_afni=func_model.cli.model_afni:main",
            "model_fsl=func_model.cli.model_fsl:main",
        ]
    },
    install_requires=[
        "numpy>=1.22.3",
        "pandas>=1.4.4",
        "setuptools>=65.5.0",
    ],
)
