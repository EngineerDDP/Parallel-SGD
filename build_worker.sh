if [ $# -ne 1 ]; then
  echo "Target Worker dir must be provided as parameter."
  echo "Usage:"
  echo -e "\t ./build_worker.sh {target directory}"
  exit 1
fi

src_files=(
  "./worker.py"
  "./constants.py"
  "./log.py"
  "./Dockerfile"
)
src_dirs=(
  "./network"
  "./nn"
  "./parallel_computing"
  "./parallel_sgd"
  "./rpc"
)

files_and_dir_check=1

for file in "${src_files[@]}"; do
  {
    if [ ! -f "$file" ]; then
      echo -e "File not found:\t\t$file"
      files_and_dir_check=0
    fi
  }
done

for dir in "${src_dirs[@]}"; do
  {
    if [ ! -d "$dir" ]; then
      echo -e "Directory not found:\t$dir"
      files_and_dir_check=0
    fi
  }
done

# Exit if some files are missing
if [ $files_and_dir_check -eq 0 ]; then
  echo "Error:"
  echo "This script can only be run inside project directory."
  exit 2
fi

worker=$1

if [ ! -d "$worker" ]; then
  mkdir "$worker"
else
  echo "Target directory exists, override ? (yes)"
  read input
  if [ ! $input = "yes" ]; then
    exit 0
  else
    echo "Override files..."
    rm -rf "${worker:?}/"*
  fi
fi



for dir in "${src_dirs[@]}"; do
  {
    cp -r $dir "$worker"/"$dir"
    echo "Copy directory: $dir"
  }
done
for file in "${src_files[@]}"; do
  {
    cp $file "$worker"/"$file"
    echo "Copy file: $file"
  }
done

python3 -m compileall "$worker"/

echo -e "\033[0;32;1mDone\033[0m"
