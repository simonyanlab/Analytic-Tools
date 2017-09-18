#!/bin/sh
# Build documentation in `gh-pages` branch

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
