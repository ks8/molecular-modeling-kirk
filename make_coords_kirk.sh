#!/bin/bash


mkdir -p data_coords_endpoints
cooling_rate="N2e7"

for i in `seq 0 10000`; do
    wdir="$cooling_rate/$i"
    index=`cut -f 2 -d '/' <<< $wdir`
    index=`printf %05d $index`

    echo $index
    ./coords_config.py $wdir/traj.atom 20000000 data_coords_endpoints/glass.${index}.txt

    ./coords_config.py $wdir/traj.atom 100000 data_coords_endpoints/liquid.${index}.txt

done
