# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "general-machine-learning"
copyright = "%Y"
author = "Matthew T Gill"
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
# https://sphinx-book-theme.readthedocs.io/en/stable/tutorials/get-started.html#add-a-source-repository-button-to-your-theme

html_theme = "sphinx_book_theme"
html_static_path = ["_static"] # these files are copied after the builtin static files, so a file named "default.css" will overwrite the builtin "default.css"
html_logo = "_static/images/logo.jpg"  # image to use as the logo above the left nav bar
html_favicon = "_static/images/favicon.ico"  # image to use for the browser tab
html_css_files = ['css/custom.css'] # custom CSS files (paths relative to html_static_path)
html_theme_options = {
    "show_toc_level": 4, # show up to heading level 5 in the right "Contents" sidebar
    "repository_url": "https://github.com/CombatWombatHub/general-machine-learning", # replace with your repository URL
    "use_repository_button": True, # create a link to the repository on the page
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