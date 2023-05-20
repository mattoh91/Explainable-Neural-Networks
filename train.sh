#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -c configFileName"
   echo -e "\t-c Hydra config filename. Eg. pipelines_resnet.yaml"
   echo -e "\t-c Description of what is parameterC"
   exit 1 # Exit script after printing help
}

while getopts "c:" opt
do
   case "$opt" in
      c ) configFileName="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$configFileName" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
python src/train.py -cn $configFileName
