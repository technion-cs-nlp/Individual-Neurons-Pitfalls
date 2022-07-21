CONVERSION_SCRIPT_DIR=ud-compatibility/UD_UM/
UD_FOLDER=../../data/UD/
UM_FOLDER=../../data/UM/
UD_LANGUAGE_PATTERN="\/([a-z]*)(_[a-zA-Z_]*)*-ud-[a-z\-]*\.conllu"
UM_LANGUAGE_PATTERN="\/([a-z]*)(_[a-zA-Z_]*)*-um-[a-z\-]*\.conllu"
orig_dir=$(pwd)
cd $CONVERSION_SCRIPT_DIR
for lan_dir in "$UD_FOLDER"/*
do
	echo $lan_dir
	for f in "$lan_dir"/*
	do
		echo $f
		if [[ $f =~ $UD_LANGUAGE_PATTERN ]]; then
			lan_code=${BASH_REMATCH[1]}
			echo "Converting $f using langauge $lan_code"
			python marry.py convert --ud $f -l lan_code
		fi
	done
	mkdir -p ${UM_FOLDER}${lan_code}
	for f in "$lan_dir"/*
	do
		if [[ $f =~ $UM_LANGUAGE_PATTERN ]]; then
			mv $f ${UM_FOLDER}${lan_code}
		fi
	done
done
cd $orig_dir

