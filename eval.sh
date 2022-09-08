set -e
export PATH=/bin:$PATH
ENVNAME=pytorch
ENVDIR=$ENVNAME

export HOME=$PWD

#export PATH
mkdir $ENVDIR
mkdir outputs
mkdir chtc
tar -xzf pytorch.tar.gz -C $ENVDIR
tar -xzf chtc.tar.gz
export PATH=$ENVDIR/bin:$PATH
. $ENVDIR/bin/activate

# echo some HTCondor job information
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "System: $(uname -spo)"
echo "_CONDOR_JOB_IWD: $_CONDOR_JOB_IWD"
echo "GPU: $(lspci | grep NVIDIA)"
echo "Cluster: $CLUSTER"
echo "Process: $PROCESS"
echo "RunningOn: $RUNNINGON"
echo "Current directory: $PWD"

python3 eval.py --start $3 --scale 1000 --source $2 --testdf chtc/spoc-train-test.tsv --beam 1

rm -rf $ENVDIR pytorch.tar.gz chtc2.tar.gz chtc/
