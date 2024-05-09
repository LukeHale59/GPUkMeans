#!/bin/bash

make -f MakefileCPU.mk
cat datasets/dataset4.txt | ./obj32/kmeansCPU
