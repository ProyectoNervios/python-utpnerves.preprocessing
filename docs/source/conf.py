# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../utpnerves'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'UTPNerves - Preprocessing'
copyright = '2022, Universidad Tecnológica de Pereira'
author = 'Universidad Tecnológica de Pereira'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'nbsphinx',
    'dunderlab.docs',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_theme_options = {
    'caption_font_family': 'Noto Sans',
    'font_family': 'Noto Sans',
    'head_font_family': 'Noto Sans',
    'page_width': '1280px',
    'sidebar_width': '300px',
}

autodoc_mock_imports = [
    'IPython',
    'matplotlib',
    'numpy',
    'utpnerves',
    'cv2',
    'tensorflow',
]

dunderlab_maxdepth = 3
# dunderlab_color_links = '#12c5a5'
dunderlab_code_reference = True
dunderlab_github_repository = 'https://github.com/ProyectoNervios/python-utpnerves.preprocessing'

latex_domain_indices = True

