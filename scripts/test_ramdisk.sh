#!/bin/bash

source ./ramdisk.sh
file=/scratches/stroustrup/jfb54/adv-fsl
ramdisk $file file

echo $file
