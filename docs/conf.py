# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.metadata, sys, os
sys.path.insert(0, os.path.abspath(".."))

project = 'dymad'
copyright = '2025, APUS Lab @ PSU'
author = 'APUS Lab @ PSU'

version = '0.1.0'
release = importlib.metadata.version("dymad")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "myst_parser",   # Loaded by myst_nb
    "myst_nb",
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    "sphinx_autodoc_typehints",
    'sphinxemoji.sphinxemoji'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

nb_execution_mode = "off"         # "auto", "off", "force", or "cache"
# nb_execution_cache_path = ".jupyter_cache"  # default; nice to keep env tidy
# nb_execution_timeout = 180        # seconds per cell (tune as needed)

myst_enable_extensions = [
    "amsmath",
    "dollarmath"
]
myst_dmath_allow_labels = True
myst_dmath_double_inline = True

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
