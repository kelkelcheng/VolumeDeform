#!/bin/bash

# ../../Downloads/rawdata/data1116/4 

for i in {6..6}
do
    ./volume_deform.out -d ../../Downloads/rawdata/moving_data/12 -n 5 -ni 3 -i 5 -s $i -dm 1 -fh 0.33 -dmt 400
    if ((i%10 == 0)); then
        sleep 1
    fi
done
