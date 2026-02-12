LATEXMK ?= latexmk
LATEXMKFLAGS ?= -pdf -interaction=nonstopmode -halt-on-error

TEX_01 := 01_Data_Preparation/01_Data_Preparation.tex
TEX_02 := 02_Run_Extraction/02_Run_Extraction.tex
TEX_03 := 03_Drag_Calculation/03_Drag_Calculation.tex
TEX_03_LINEAR := legacy/03_Drag_Calculation_Linear_in_v.tex
TEX_MAIN := main.tex

.PHONY: all docs tables clean

all: docs

tables:
	python scripts/generate_drag_report_tables.py

docs: tables
	$(LATEXMK) $(LATEXMKFLAGS) $(TEX_01)
	$(LATEXMK) $(LATEXMKFLAGS) $(TEX_02)
	$(LATEXMK) $(LATEXMKFLAGS) $(TEX_03)
	$(LATEXMK) $(LATEXMKFLAGS) $(TEX_03_LINEAR)
	$(LATEXMK) $(LATEXMKFLAGS) $(TEX_MAIN)

clean:
	$(LATEXMK) -C $(TEX_01)
	$(LATEXMK) -C $(TEX_02)
	$(LATEXMK) -C $(TEX_03)
	$(LATEXMK) -C $(TEX_03_LINEAR)
	$(LATEXMK) -C $(TEX_MAIN)
