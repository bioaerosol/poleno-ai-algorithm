#!/bin/sh

set -e
set -u


Poleno_hourly_zips=/data/input/
Poleno_output=/data/output/
Log_dir=/data/logs/
Poleno_scripts=/wd/src/scripts

mkdir -p $Poleno_output

for Zip_to_process in `find $Poleno_hourly_zips -name "*.zip" -printf "%f\n"`; do

	filename=$Zip_to_process
	base_name=$(basename "$filename" .${filename##*.})
	substring=$(echo "$base_name" | rev | cut -c 1-14 | rev)
	Zip_date=`echo "$substring"`

	#Zip_date=`echo $Zip_to_process | cut -b 1-14`
	
	echo 'Processing ' $Zip_to_process
	Temp_folder=src/temp/$Zip_to_process

	start=$(date +%s)
	if [ -d "$Temp_folder" ]; then rm -Rf $Temp_folder; fi
	mkdir $Temp_folder -p 
	echo "Unzipping file to temp folder... "

	unzip -q $Poleno_hourly_zips/$Zip_to_process -d $Temp_folder
	end=$(date +%s)
	echo 'Time elapsed for' $Zip_to_process 'preparation:' $(($end-$start)) 'seconds'
	

	start=$(date +%s)
	EXIT_CODE='none'

	echo 'Running the algorithm and creating JSON... '
	
	python $Poleno_scripts/Recognition_11_classes_operational.py $Temp_folder $Poleno_output $Zip_date || EXIT_CODE=$?
	echo 'Errors: ' $EXIT_CODE
	
	end=$(date +%s)
	echo 'Time elapsed for' $Zip_to_process 'processing:' $(($end-$start)) 'seconds'
	

	start=$(date +%s)
	echo 'Deleting temp folder...'
	
	rm -Rf $Temp_folder
	end=$(date +%s)
	echo 'Time elapsed for cleaning:' $(($end-$start)) 'seconds'
	
done
