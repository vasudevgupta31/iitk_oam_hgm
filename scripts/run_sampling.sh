#!/bin/bash

CONFIGFILE=$1

if [ $# -eq 0 ] ; then
    echo "Configfile path not supplied."
else

# We allow a maximum of four jobs
# to run in parallel to avoid using
# too much resources
declare pids=( )
num_procs=4

while (( ${#pids[@]} >= num_procs )); do
    #sleep is not a "clean" option, but
    #old version of bash don't support wait -n
    sleep 0.2
    for pid in "${!pids[@]}"; do
        kill -0 "$pid" &>/dev/null || unset "pids[$pid]"
    done
done
python do_sampling.py --configfile $CONFIGFILE --verbose True & pids["$!"]=1

wait

python do_analysis.py --configfile $CONFIGFILE --verbose True
fi