#!/bin/bash

./runTensorFlowLIDC.py 2>&1 | tee output/`date +%Y%m%d-%H%M%S`-runTensorFlowLIDC.out.txt