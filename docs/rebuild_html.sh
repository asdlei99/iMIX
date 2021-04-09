#!/bin/bash
#set -v
cd ~/code/imix/docs
read -p "please check which condiction you are: if you just add some annotation and your code was not changed, please input 1; if you add or delete the **.py files, please input 2..." input
if [ -z $input ]; then
	echo "invalid input...."
	exit
else
	if [ $input -eq 1 ];then
		make clean
		make html
		exit
	else
		if [ $input -eq 2 ];then
			echo -e "\033[41;33m please delete the rst file corresponding to directory of the <**.py> file\033[0m"
			read -p "if you have already finished the previous step, then input 3..." input
			if [ $input -eq 3 ];then
				make clean
				sphinx-apidoc -o source ../imix/
				make html
				exit
			else
				echo "invalid input...."
				exit
			fi
		fi
	fi
fi
	#	else
		#	echo "there didn't exist any other cases...."
		#	exit
