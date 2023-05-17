.PHONY: mkdocs-serve mkdocs-build mkdocs-clean

mkdocs-clean:
	rm site/

mkdocs-serve:
	@echo "Serving nos docs"
	mkdocs serve

mkdocs-build:
	@echo "Building nos docs"
	mkdocs build
