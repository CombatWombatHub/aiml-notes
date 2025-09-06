# Example Markdown File

## Sphinx Placeholder
- just making this as a placeholder to test basic documentation building

## Mermaid Example
- Sphinx can build Mermaid diagrams thanks to `sphinxcontrib.mermaid` and `myst-parser` (included with `myst-nb`)
- if `myst_fence_as_directive = ["mermaid"]` is set in `conf.py`, Sphinx will render diagrams created with triple backtick mermaid code blocks (otherwise you'll get the error `Pygments lexer name 'mermaid is not known` while building docs)

```mermaid
graph LR
    A-->B
    B-->C
```