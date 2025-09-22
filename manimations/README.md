# Manim
- Python library for making animations, created by 3Blue1Brown, now community-maintained.
- per the [installation instructions](https://docs.manim.community/en/stable/installation/uv.html):
    - install `manim` into the project with `uv` (done - it will be installed if you run `uv install --all-groups`)
    - install `LaTeX` if you want to add equations to your graphics (I do) - For Windows, they recommend the [MiKTeX distribution](https://miktex.org/) from https://miktex.org/download (this step must be done on any new computer I want to use `manim` on. They have other recommendations for `MacOS` and `Linux`).
    - run `uv run manim checkhealth` to check if your installation is good to go with `manim`. I had to restart `VSCode` for it to pick up the installation, and when I ran it, it said everything had passed and then `VSCode` crashed.
- [quickstart guide](https://docs.manim.community/en/stable/tutorials/quickstart.html)
    - interestingly, they recommend using this to import everything:
        - `from manim import *`
    - This runs counter to my normal Python expectations, but maybe you end up using so many names that it ends up being worth it
        - on second thought, just do imports the normal way so MyPy doesn't freak out on me

## Sphinx
- The video files created by `manim` are going into the `media` directory by default
- can embed local video files to Sphinx with [sphinxcontrib-video](https://sphinxcontrib-video.readthedocs.io/en/latest/quickstart.html)
    - I am hesitant to store videos in the repo itself
    - could potentially generate them in the pipeline straight into the `_static` directory
    - unsure if GitHub would flag that as files being too big though
- can embed videos hosted on `YouTube`, `Vimeo`, `Peertube` files in Sphinx with [sphinxconrib-youtube](https://sphinxcontrib-youtube.readthedocs.io/en/latest/usage.html)
    - could upload my videos to a remote location like this and reference them with links in the documentation