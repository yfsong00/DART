dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="cora"
[ -z "${device}" ] && device=-1


python -m pdb link_prediction.py \
	--device $device \
	--dataset $dataset 
