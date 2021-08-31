#!/bin/bash

# specify variables ####
temp=~/scratch60/scnd/data/raw/human/
to_gdrive=False
tape_fpath=/SAY/archive/YCGA-729009-YCGA/archive/pacbio/gw92/10x/Single_Cell/
file1=${tape_fpath}20200707_llt26.tar
file2=${tape_fpath}20200223_llt26.tar
# ######################

# make temp directory
mkdir -p ${temp}
for file in {$file1,$file2}
do
  cp $file ${temp}
  printf 'copied %s from tape' $(basename $file)
done

if [[ $backup=True ]]
then
  module load Rclone
  rclone copy ${temp}* gdrive:temp/tejwani_ravindra_etal2020/
  echo "done with transfer to temp gdrive"
fi

# untar
cd ${temp}
for tarfile in ${temp}*.tar
do
  tar -xvf $tarfile
  printf 'done unzipping %s' $tarfile
done
  
exit

