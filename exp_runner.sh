py_file=`realpath $1`
export config_file=`realpath $2`
dataset=$3


cd `dirname $py_file`
if [ -z "$dataset" ]
then
    echo "skip preparing dataset"
else
    bash ./prepare_data.sh $dataset
fi
export PYTHONPATH=$PYTHONPATH:./qsparse-private:.
CUDA_VISIBLE_DEVICES=0 python3.8 $py_file

