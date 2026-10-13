# make tables figures paper   (uses the result files in cutting_sim/output; re-run experiments first, see REPRODUCE.md)
PY ?= python
.PHONY: tables figures paper test clean
tables:
	cd paper && $(PY) make_tables.py
figures:
	cd paper && OPENBLAS_NUM_THREADS=1 $(PY) make_figures.py
paper: tables figures
	cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
test:
	cd cutting_sim && OPENBLAS_NUM_THREADS=1 $(PY) test_frf_import.py
clean:
	cd paper && rm -f main.aux main.bbl main.blg main.log main.out
