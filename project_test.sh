# check directory
test_root="integration_tests"
if [ ! -d $test_root ]; then
    echo -e "[\033[0;92;1mERROR\033[0m]Please run this script inside project root."
    exit 1
fi

test_cases=(
    "rpc_test.py"
    "ssgd_dnn_test.py"
    "nn_test.py"
    "network_test.py"
)

# backup env
env_bak=$PYTHONPATH
# set enviroment
export PYTHONPATH=$(pwd)

# run all tests
for test in "${test_cases[@]}"; do
  {
    echo -e "\033[0;32;1mRunning\033[0m: $test" 
    python3 $test_root/$test
  }
done

# recover env
export PYTHONPATH=$env_bak
