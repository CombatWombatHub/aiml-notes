# Building Sphinx

## Locally On Windows
- prerequisites
    - [install uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) (I did `curl -LsSf https://astral.sh/uv/install.sh | sh` in `git bash`)
    - [install python](https://docs.astral.sh/uv/guides/install-python/#getting-started) (I did `uv python install`)
    - [install dependencies](https://docs.astral.sh/uv/guides/projects/#running-commands) with `uv sync --group docs` to install Python packages like `sphinx`, `myst-parser` etc
    - [install pandoc](https://pandoc.org/installing.html) (e.g. download & run `pandoc-<version>-windows-x86_64.msi`) and restart `VSCode` so that it's in `PATH`
- running
    - Open `Git Bash` terminal
    - `cd` to `docs` directory
    - run `./make.bat html`
- opening
    - the previous steps will create the documentation in [docs/build/html](./build/html/index.html)
    - launch that


## GitHub Pages
- pushing on `main` will build Sphinx documentation and push to GitHub Pages
- this is due to the GitHub Actions from the job specified in [sphinx.yml](../.github/workflows/sphinx.yml)
- if the pipeline is successful, it should build and deploy an [updated version of the documentation on GitHub Pages](https://combatwombathub.github.io/general-machine-learning/index.html)
