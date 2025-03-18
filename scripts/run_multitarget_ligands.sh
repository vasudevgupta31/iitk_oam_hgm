#!/bin/bash

CONFIGFILE=$1

if [ $# -eq 0 ] ; then
    echo "Configfile path not supplied."
else

bash run_processing.sh $CONFIGFILE &&
bash run_pooled_training.sh $CONFIGFILE &&
bash run_beam_generation.sh $CONFIGFILE &&
echo "Use the beam search designs to check which epochs are to be sampled. Then start: bash run_sampling.sh configfiles/CONFIGFILE."

fi