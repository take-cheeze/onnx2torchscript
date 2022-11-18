from setuptools import setup, find_packages

setup(
    name='onnx2torchscript',
    version='0.0.1',
    packages=find_packages(include=['onnx2torchscript']),
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
