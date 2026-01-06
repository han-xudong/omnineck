#!/bin/bash

pasway=$(dirname "$0")
cd "$pasway" || exit 1

for file in "$pasway"/*.inp; do
    jobname=$(basename "$file" .inp)
    echo "Running Abaqus job: $jobname"

    rm -f "${jobname}".{odb,log,lck,com,dat,prt,sta,msg,res}

    abq job="$jobname" cpus=4 int
done

echo "All finished."
read -n 1 -s -r -p "Press any key to continue..."
echo
