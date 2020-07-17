#!/bin/bash
#The function takes 3 parameters,
#the first the file to add to ram disk
#the second is a variable to place the new file path into
#the thirds is whether to update the contents if they have changed (optional, default false)
set -x 
function ramdisk {
    local __file=$1
    local __ramdiskFolder=/tmp/ramdisk/$(basename -- $__file)
    #if the folder exists, and it does not contain the file already, scrap the ramdisk and start over
    if [[ $3 == 'true' && -d $__ramdiskFolder && !  $(diff $__ramdiskFolder/$(basename -- $__file) $__file) ]]; then
        echo "ramdisk exists, but has the wrong contents"
        sudo umount $__ramdiskFolder
        rm -r $__ramdiskFolder
    fi
    #if the folder does not exist, create it
    if [[ ! -d $__ramdiskFolder ]]; then
        echo "$__ramdiskFolder does not exist, creating"
        mkdir -p $__ramdiskFolder
        chmod 777 $__ramdiskFolder
        sudo mount -t tmpfs -o size=$(du -B 1 -s  $__file | cut -f 1) myramdisk $__ramdiskFolder
	rsync --info=progress -auvz $__file $__ramdiskFolder

    fi

    local  __resultvar=$2
    local  myresult=$__ramdiskFolder/$(basename -- $__file)
    eval $__resultvar="'$myresult'"
}
