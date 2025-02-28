#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_dynamic_cluster.sh"
echo "==========================================="

EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

HOSTFILE=$1
mpirun -n 16 --hostfile $HOSTFILE --output-filename log_output \
    --merge-stderr-to-stdout python net.py
