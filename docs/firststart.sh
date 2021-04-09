#!/bin/bash
set -v
cd ~/code/imix/
read "please make sure you are in the project directory <imix>:? [Y/n]" input
conda activate IMIX
pip install sphinx
pip install recommonmark
pip install sphinx_markdown_tables
pip install sphinx_rtd_theme
read "please make sure your source code are in the sub directory <imix>:? [Y/n]" input
read "please make sure you have the directory <docs>:? [Y/n]" input
cd docs
sphinx-apidoc -o source ../imix/
make html
