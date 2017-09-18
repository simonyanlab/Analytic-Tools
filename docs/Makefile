# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = AnalyticTools
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom clean directive that purges not only the build folder but also all auto-generated stub *.rst files
clean:
	rm -rf $(BUILDDIR)/
	rm -rf $(SOURCEDIR)/_stubs/

# Build documentation in `gh-pages` branch
gh-pages:
	cd ..
	git checkout gh-pages
	rm -rf _images _modules _sources _static _stubs
	git checkout master docs *.py *.pyx
	git reset HEAD
	cd docs/
	make html
	cd ..
	mv -fv docs/build/html/* ./
	rm -rf docs *.py *.pyc *.pyx
	git add -A
	git commit -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`" && git push origin gh-pages ; git checkout master
