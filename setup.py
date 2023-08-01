import os

from setuptools import find_packages, setup

with open('version.txt', "r") as file_handler:
    __version__ = file_handler.read().strip()


long_description = """

""" 


setup(
    name="morph_pre",
    packages=find_packages(),
    package_data={"morph_pre": ["py.typed", "version.txt"]},
    install_requires=[
    ],
    
    description="",
    author="Jiaming Guo",
    url="",
    author_email="",
    license="MIT",
    version=__version__,
)