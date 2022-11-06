from setuptools import setup, find_packages

setup(
    name='onnx2pytorch',
    version='0.0.1',
    packages=find_packages(include=['onnx2pytorch']),
    install_requires=[
        'onnx',
        "pytorch-pfn-extras",
        'torch',
    ],
    extras_require={
        "test": [
            "pytest",
            "tabulate",
        ],
    },
)
