ICDSA 2026 / Springer LNNS -- LaTeX source package
"Early Detection of Model Collapse in Large Language Models: A Diversity-Based Framework"

BUILD (default, no BibTeX needed):
    pdflatex main
    pdflatex main          (run twice so cross-references resolve)

The references are embedded as a thebibliography block, so no external .bst is
required. If you prefer BibTeX with the official Springer style, in main.tex
replace the thebibliography block with:
    \bibliographystyle{splncs04}
    \bibliography{references}
and build with:  pdflatex main ; bibtex main ; pdflatex main ; pdflatex main

FILES
  main.tex         Main LaTeX source (Springer svproc class). Option B.
  references.bib   BibTeX entries (optional).
  svproc.cls       Springer proceedings class (from the official template).
  figure1.pdf      Figure 1 (vector PDF).
  figure1.png      Figure 1 (raster fallback).
  main.pdf         Pre-compiled reference output.

NOTES
  Add your ORCID in the \author/\institute block if you have one
  (0009-0006-9706-1572). Confirm the GitHub URL in Section 1.
