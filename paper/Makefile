FILE=main
FILE_2=main_v1
default:
	pdflatex $(FILE).tex

final:
	pdflatex $(FILE).tex
	bibtex $(FILE)
	pdflatex $(FILE).tex
	pdflatex $(FILE).tex

diff:
	latexdiff $(FILE_2).tex $(FILE).tex > diff.tex
	pdflatex diff.tex
	bibtex diff
	pdflatex diff.tex
	pdflatex diff.tex

clean:
	rm *~ *.aux *.out *.toc *.blg *.bbl *.log *.tdo *.dvi
