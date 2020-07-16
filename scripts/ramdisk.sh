#!/bin/bash
#The function takes 3 parameters,
#the first the file to add to ram disk
#the second is a variable to place the new file path into
#the thirds is whether to update the contents if they have changed (optional, default false)
function ramdisk {
    local __file=$1
    #if the folder exists, and it does not contain the file already, scrap the ramdisk and start over
    if [[ $3 == 'true' && -d /tmp/ramdisk/ && !  $(diff /tmp/ramdisk/$(basename -- $__file) $file) ]]; then
        echo "ramdisk exists, but has the wrong contents"
        sudo umount /tmp/ramdisk/
        rm -r /tmp/ramdisk/
    fi
    #if the folder does not exist, create it
    if [[ ! -d /tmp/ramdisk ]]; then
        echo "/tmp/ramdisk/ does not exist, creating"
        mkdir /tmp/ramdisk/
        chmod 777 /tmp/ramdisk/
        sudo mount -t tmpfs -o size=$(du -B 1  $__file | cut -f 1) myramdisk /tmp/ramdisk
        cp -r $file /tmp/ramdisk/
    fi

    local  __resultvar=$2
    local  myresult=/tmp/ramdisk/$(basename -- $__file)
    eval $__resultvar="'$myresult'"
}