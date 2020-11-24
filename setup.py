import setuptools

setuptools.setup(
    name="ParallelSGDToolkit",
    version="0.75",
    author="Engineer_DDP",
    descriptions="An all in one toolkit package for Parallel-SGD experiments.",
    url="https://github.com/EngineerDDP/Parallel-SGD",
    packages=setuptools.find_packages(),

    classifiers={
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: win64",
    },

    python_requires=">=3.6"
)
