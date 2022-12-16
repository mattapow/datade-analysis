#!/bin/bash
# run the same liggghts simulations accross multiple slope angles

cd ../Inclined_deep

# run in N parallel processes
N=5

(
for theta in {23..27}; do
    ((i=i%N)); ((i++==0)) && wait
    # add half a degree to the angles
    angle=$(echo $theta + 0.5 | bc)
    path=$(echo theta$angle)
    . "../liggghts_single.sh" $path $angle &
done
)
echo "done"
