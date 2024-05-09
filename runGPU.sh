#!/bin/bash

make -f MakefileGPU.mk
cat datasets/dataset4.txt | ./obj32/kmeansGPU
