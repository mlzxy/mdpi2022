py_file=`realpath $1`
export config_file=`realpath $2`
dataset=$3
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd `dirname $py_file`
if [ -z "$dataset" ]
then
    echo "skip preparing dataset"
else
    bash ./prepare_data.sh $dataset
fi

use_public_qsparse=${USE_PUBLIC_QSPARSE:-0}
if [ "$use_public_qsparse" -eq "0" ]; then
    export PYTHONPATH=$SCRIPT_DIR/qsparse-private:$SCRIPT_DIR
else
    export PYTHONPATH=$SCRIPT_DIR/qsparse-public:$SCRIPT_DIR
fi

CUDA_VISIBLE_DEVICES=0 python3.8 $py_file

