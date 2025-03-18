#!/bin/bash

CONFIGFILE=$1

if [ $# -eq 0 ] ; then
    echo "Configfile path not supplied."
else

python do_beam_search.py --configfile $CONFIGFILE --verbose True

fi