#!/bin/bash
export LD_LIBRARY_PATH=/cs/local/lib/pkg/cudatoolkit/lib64:../matio:$LD_LIBRARY_PATH
./minres ../matrices/maxwell1.mat 1