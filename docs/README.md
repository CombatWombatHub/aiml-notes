# Building Sphinx

## Locally On Windows
- prerequisites
    - [install uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) (I did `curl -LsSf https://astral.sh/uv/install.sh | sh` in `git bash`)
    - [install python](https://docs.astral.sh/uv/guides/install-python/#getting-started) (I did `uv python install`)
    - [install dependencies](https://docs.astral.sh/uv/guides/projects/#running-commands) with `uv sync --group docs` to install Python packages like `sphinx` and `myst-nb` (note: [myst-nb](https://myst-nb.readthedocs.io/en/latest/) replaced `nbsphinx`, which required a separate install of [pandoc](https://pandoc.org/installing.html). It also already includes [myst-parser](https://myst-parser.readthedocs.io/en/latest/), so you don't need to install or activate it separately)
- running
    - Open `Git Bash` terminal
    - `cd` to `docs` directory
    - run `./make.bat html`
- note
    - sometimes this doesn't update already-built pages
    - in this case, run `./make.bat clean` first
    - or run `./clean-make.bat` which runs both `.make.bat clean` and `.make.bat html`
- opening
    - the previous steps will create the documentation in [docs/build/html](./build/html/index.html)
    - launch that


## Deploy to GitHub Pages via GitHub Actions Pipeline
- pushing on `main` will build Sphinx documentation and push to GitHub Pages
- this is due to the GitHub Actions from the job specified in [sphinx.yml](../.github/workflows/sphinx.yml)
- if the pipeline is successful, it should build and deploy an [updated version of the documentation on GitHub Pages](https://combatwombathub.github.io/general-machine-learning/index.html)
