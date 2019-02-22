from os import path
from setuptools import setup, find_packages
import sys

if sys.version_info < (3, 5, 0):
    typing = ["typing"]
else:
    typing = []

# Get the long description from the README file
with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="symspell",
    description="PySymSpell is a pure Python port of SymSpell.",
    long_description=long_description,
    version="0.1.0",
    license="MIT",
    url="https://github.com/ne3x7/pysymspell",
    download_url="https://github.com/ne3x7/pysymspell",
    packages=find_packages(),
    keywords=["spelling", "spell checker", "symspell"],
)
