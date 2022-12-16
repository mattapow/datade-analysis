cd ../Inclined_deep

for theta in {27..27}
do
    # angle=$(echo $theta + 0.5 | bc)
    angle=$theta
    echo $angle
    post process in dump folder
    cd $HOME/Documents/datade/Inclined_deep/theta$angle/dump
    PostProcessing --dump2vtk --W '[vtk]def,radius,vel' testinclined_deep

    # coarse grain from experiment folder.
    # Use same generic coarse graining json file.
    cd ..
    CoarseGraining ../CG.json
done
