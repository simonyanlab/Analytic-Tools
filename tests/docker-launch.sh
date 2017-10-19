#!/bin/sh
# Launch Docker container for unit-testing

docker exec -it --user travis travis-debug-at bash -l
