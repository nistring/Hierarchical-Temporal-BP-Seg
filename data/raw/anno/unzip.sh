#!/bin/sh
for zip in *.zip
do
  dirname=`echo $zip | sed 's/\.zip$//'`
  if mkdir "$dirname"
  then
    if cd "$dirname"
    then
      unzip ../"$zip"
      cd ..
      # rm -f $zip # Uncomment to delete the original zip file
    else
      echo "Could not unpack $zip - cd failed"
    fi
  else
    echo "Could not unpack $zip - mkdir failed"
  fi
done

# for i in $(seq 1 75)
# do
#   old_name=$(printf "CNUH_DC04_BPB1_%04d.mp4.zip" $i)
#   new_name=$(printf "CNUH_DC04_BPB1_%04d.zip" $i)
#   if [ -f "$old_name" ]
#   then
#     mv "$old_name" "$new_name"
#   else
#     echo "File $old_name does not exist"
#   fi
# done