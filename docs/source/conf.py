# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'UTPNerves - Preprocessing'
copyright = '2022, author'
author = 'author'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',

    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',

    'nbsphinx',
    'sphinx.ext.mathjax',

    'sphinxcontrib.bibtex',
]

naoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_theme_options = {
    'page_width': '1280px',
    'sidebar_width': '300px',

    # 'fixed_sidebar': True,

    # 'show_relbars': True,
    # 'show_relbar_bottom': True,


    # 'github_user': 'bitprophet',
    # 'github_repo': 'alabaster',
}

autodoc_mock_imports = [
    'IPython',
    'matplotlib',
    'numpy',
]

todo_include_todos = True


def setup(app):
    app.add_css_file("custom.css")


highlight_language = 'none'
html_sourcelink_suffix = ''


nbsphinx_execute = 'never'
# nbsphinx_input_prompt = ' '
# nbsphinx_output_prompt = ' '
nbsphinx_kernel_name = 'python3'
nbsphinx_prompt_width = '0'

nbsphinx_prolog = """
.. raw:: html

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
    }
    </style>
"""


notebooks_dir = 'notebooks'

notebooks_list = os.listdir(os.path.join(
    os.path.abspath(os.path.dirname(__file__)), notebooks_dir))
notebooks_list = filter(lambda s: not s.startswith('__'), notebooks_list)

notebooks = []
for notebook in notebooks_list:
    if notebook not in ['readme.ipynb', 'license.ipynb'] and notebook.endswith('.ipynb'):
        notebooks.append(f"{notebooks_dir}/{notebook.replace('.ipynb', '')}")

notebooks = '\n   '.join(sorted(notebooks))

with open('index.rst', 'w') as file:
    file.write(f"""
.. include:: {notebooks_dir}/readme.rst

.. toctree::
   :maxdepth: 2
   :name: mastertoc

   {notebooks}


.. only:: html

    Docstrings
    ==========

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`

    """)

bibtex_bibfiles = ['refs.bib']


