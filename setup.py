from setuptools import setup, find_packages

setup(
    name="func_model",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "func_model=func_model.entrypoint:main",
            "model_afni=func_model.cli.model_afni:main",
        ]
    },
)
