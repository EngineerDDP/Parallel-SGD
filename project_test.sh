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
# set environment
PYTHONPATH=$(pwd)
export PYTHONPATH

passed_test=0
failed_test=0

# run all tests
for test in "${test_cases[@]}"; do
  {
    echo -e "\033[0;32;1mRunning\033[0m: \033[0;93;1m$test\033[0m"

    python3 -m unittest $test_root/$test > /dev/null

    if [ $? -eq 0 ]; then
      echo -e "\033[0;32;1mTest Passed\033[0m: \033[0;93;1m$test\033[0m"
      passed_test=$[ $passed_test+1 ]
    else
      echo -e "\033[0;31;1mTest Failed\033[0m: \033[0;93;1m$test\033[0m"
      failed_test=$[ $failed_test+1 ]
    fi
  }
done

if [ -d "tmp_log" ]; then
  echo -e "\033[0;34;1m Clear environment.\033[0m"
  rm -rf tmp_log
fi

echo -e "\033[0;32;1mTest Complete\033[0m \033[0;32;1m$passed_test Passed\033[0m, \033[0;31;1m$failed_test Failed\033[0m"

# recover env
export PYTHONPATH=$env_bak
