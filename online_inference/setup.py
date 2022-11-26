from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="My first ml project",
    author="Masaeva Olga, VK-Made, MLE-12",
    entry_points={
        "console_scripts": [
            "ml_train = train_pipeline:train_pipeline_command"
        ]
    },
    install_requires=required,
    license="MIT",
)