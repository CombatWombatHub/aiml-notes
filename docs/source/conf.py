# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "aiml-notes"
copyright = "%Y"
author = "Matthew T Gill"
release = "1.0"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",  # parse Jupyter Notebooks, includes myst_parser, replaces nbsphinx, doesn't need separate pandoc install
    "sphinxcontrib.mermaid", # allow mermaid diagrams
    "sphinx_design", # allow arranging elements with tabs, etc.
]

templates_path = ["_templates"]

# exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://sphinx-book-theme.readthedocs.io/en/stable/tutorials/get-started.html#add-a-source-repository-button-to-your-theme

html_theme = "sphinx_book_theme"
html_static_path = ["_static"] # these files are copied after the builtin static files, so a file named "default.css" will overwrite the builtin "default.css"
html_logo = "_static/logo.jpg"  # image to use as the logo above the left nav bar
html_favicon = "_static/favicon.ico"  # image to use for the browser tab
html_css_files = ['custom.css'] # custom CSS files (paths relative to html_static_path)
html_theme_options = {
    "show_toc_level": 4, # show up to heading level 5 in the right "Contents" sidebar
    "repository_url": "https://github.com/CombatWombatHub/aiml-notes", # replace with your repository URL
    "use_repository_button": True, # create a link to the repository on the page
    "home_page_in_toc": True, # include the home page in the left sidebar
    #"show_navbar_depth": 4, # show up to heading level 5 in the left sidebar
    #"collapse_navbar": True, # collapse the left sidebar to show only top-level headings
}


# -- Options for MyST-Parser -------------------------------------------------
#https://myst-parser.readthedocs.io/en/latest/index.html

# enable syntax extensions for MyST-Parser 
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#syntax-extensions
myst_enable_extensions = [
    #"amsmath",
    #"attrs_inline",
    "colon_fence", # allow ::: triple colons as fenced code blocks
    #"deflist",
    "dollarmath", # render $dollar sign enclosed$ equations
    #"fieldlist",
    #"html_admonition",
    "html_image", # allow HTML <img> tags
    #"linkify",
    #"replacements",
    #"smartquotes",
    #"strikethrough",
    #"substitution",
    #"tasklist",
]

# generate anchors so you can link to markdown headings
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#auto-generated-header-anchors
myst_heading_anchors = 3

# Configure myst-parser to treat mermaid fenced code blocks as directives
myst_fence_as_directive = ["mermaid"]


# -- Options for MyST-NB -------------------------------------------------
# https://myst-nb.readthedocs.io/en/v0.9.0/use/execute.html

# only execute notebooks that are missing at least one output
jupyter_execute_notebooks = "auto"

# do not execute notebooks that do computationally intensive training
execution_excludepatterns = [
    "nlp_beginners_guide.ipynb",
    "*_noex.ipynb", # "no execute" - add to notebook names to not re-run
]