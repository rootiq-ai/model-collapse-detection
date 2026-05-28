# Paper Source

LaTeX source for the ICDSA 2026 / Springer LNNS submission.

## Build

The bibliography is embedded inline (`thebibliography`), so no BibTeX run is needed:

```bash
pdflatex icdsa2026_model_collapse
pdflatex icdsa2026_model_collapse        # run twice so cross-refs resolve
```

If you prefer BibTeX with the official Springer style (Overleaf has `splncs04.bst`
preinstalled), in the .tex file replace the `thebibliography` block with:

```latex
\bibliographystyle{splncs04}
\bibliography{references}
```

and build with `pdflatex → bibtex → pdflatex → pdflatex`.

## Files
- `icdsa2026_model_collapse.tex` — main source
- `references.bib` — BibTeX (optional; only needed if you switch to the BibTeX path)
- `svproc.cls` — Springer proceedings class (from the official template)
- `figure1.pdf` / `figure1.png` — Figure 1 (LaTeX picks figure1.pdf)
