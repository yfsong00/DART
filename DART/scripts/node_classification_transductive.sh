dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="cora"
[ -z "${device}" ] && device=-1


python -m pdb node_classification_transductive.py \
	--device $device \
	--dataset $dataset 
