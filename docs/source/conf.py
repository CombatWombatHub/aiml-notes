# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "general-machine-learning"
copyright = "2025, Matthew Thomas Gill"
author = "Matthew Thomas Gill"
release = "1.0"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # parse markdown documents
    "nbsphinx",  # parse Jupyter Notebooks
    "sphinxcontrib.mermaid", # allow mermaid diagrams
]

templates_path = ["_templates"]
# exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "show_toc_level": 4 # show up to heading level 5 in the right "Contents" sidebar
}


# -- Options for MyST-Parser -------------------------------------------------
#https://myst-parser.readthedocs.io/en/latest/index.html

# enable syntax extensions for MyST-Parser 
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#syntax-extensions
myst_enable_extensions = [
    "dollarmath", # render $dollar sign enclosed$ equations
    "colon_fence", # allow ::: triple colons as fenced code blocks
]

# generate anchors so you can link to markdown headings
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#auto-generated-header-anchors
myst_heading_anchors = 3

# Configure myst-parser to treat mermaid fenced code blocks as directives
myst_fence_as_directive = ["mermaid"]