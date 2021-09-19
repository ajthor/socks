#!/bin/bash

jupyter nbconvert --exec --to latex --template ieee.tplx --output ../results/editorial.tex editorial.ipynb
cd ../results
cp ../code/references.bib .
pdflatex editorial
bibtex editorial
pdflatex editorial
pdflatex editorial
rm -f *.aux *.out *.log *.bbl *.blg *.bib
