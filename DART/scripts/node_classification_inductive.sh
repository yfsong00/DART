dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="flickr"
[ -z "${device}" ] && device=-1


python -m pdb node_classification_inductive.py \
	--device $device \
	--dataset $dataset