from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyqoi",
    version="0.1.0",
    author="Prakash S",
    author_email="prakashsellathurai@gmail.com",
    description="A Python implementation of the QOI (Quite OK Image) format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prakashsellathurai/pyqoi",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.21.0",
    ],
)