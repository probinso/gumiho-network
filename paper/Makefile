default:report.pdf

report.pdf: report.tex references.bib ../*.py
	pdflatex report.tex
	bibtex report
	pdflatex report.tex
	pdflatex report.tex

clean:
	rm -f report.{log,aux,bbl,blg}
	rm -f *~

veryclean: clean
	rm -f report.pdf

archive:default clean
	tar -cjvf philip_robinson.tar.bz2 *
