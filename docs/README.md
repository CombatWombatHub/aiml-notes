# Documentation

## Building Documentation Locally On Windows
- prerequisites
    - [install uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) (I did `curl -LsSf https://astral.sh/uv/install.sh | sh` in `git bash`)
    - [install python](https://docs.astral.sh/uv/guides/install-python/#getting-started) (I did `uv python install`)
    - create a virtual environment in `.venv` with `uv venv`
    - [install dependencies](https://docs.astral.sh/uv/guides/projects/#running-commands) with `uv sync --all-groups` to install all dependencies or `uv sync --group docs` to just install Python packages like `sphinx` and `myst-nb` (note: [myst-nb](https://myst-nb.readthedocs.io/en/latest/) replaced `nbsphinx`, which required a separate install of [pandoc](https://pandoc.org/installing.html). It also already includes [myst-parser](https://myst-parser.readthedocs.io/en/latest/), so you don't need to install or activate it separately)
- running
    - Open `Git Bash` terminal
    - `cd` to `docs` directory
    - run one of the batch files to build
        - `./make.bat clean` can clean out the previous `_build` directory
        - `./make.bat html` will build new pages, but may not update previously built pages
        - `./clean_make.bat` runs both of the above things
        - `./clean_make_launch.bat` runs both then launches the newly-built pages
    - if you get `'sphinx-build' command was not found ...`, activate your virtual environment first with
        - `source ../.venv/Scripts/activate` 
- opening
    - the previous steps will create the documentation in [docs/build/html](./build/html/index.html)
    - launch that file

## Deploying Documentation to GitHub Pages via GitHub Actions Pipeline
- pushing on `main` will build Sphinx documentation and push to GitHub Pages
- this is due to the GitHub Actions from the job specified in [sphinx.yml](../.github/workflows/sphinx.yml)
- if the pipeline is successful, it should build and deploy an updated version of the documentation on GitHub Pages at https://combatwombathub.github.io/aiml-notes/.