#!/bin/bash

# mpiexec python3 run-from-yaml.py make-cl-tz-150-symmetry.yaml
# mpiexec python3 run-from-yaml.py make-cl-tz-90-symmetry.yaml
echo "using config file $1";
mpiexec python3 make-cl-tz.py $1;