if [ $# -ne 1 ]; then
  echo "Target Worker dir must be provided as parameter."
  echo "Usage:"
  echo -e "\t ./build_worker.sh {target directory}"
  exit 1
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
    rm -rf "${worker:?}/"*
  fi
fi

if [ ! -d "$worker"/codec ]; then
  mkdir "$worker"/codec
fi
if [ ! -d "$worker"/dataset ]; then
  mkdir "$worker"/dataset
fi
if [ ! -d "$worker"/executor/psgd ]; then
  mkdir -p "$worker"/executor/psgd
fi

echo "Override files..."

cp ./codec/interfaces.py "$worker"/codec/
cp ./codec/__init__.py "$worker"/codec/
cp ./codec/essential.py "$worker"/codec/
cp -r ./dataset/transforms "$worker"/dataset/
cp ./dataset/interfaces.py "$worker"/dataset/
cp ./executor/psgd/net_package.py "$worker"/executor/psgd/
cp ./executor/__init__.py "$worker"/executor/
cp ./executor/abstract.py "$worker"/executor/
cp ./executor/interface.py "$worker"/executor/
cp -r ./models "$worker"/
cp -r ./network "$worker"/
cp -r ./nn "$worker"/
cp -r ./utils "$worker"/
cp -r ./psgd "$worker"/
cp -r ./roles "$worker"/
cp ./worker.py "$worker"/
cp ./constants.py "$worker"/
cp -r ./profiles "$worker"/

python3 -m compileall "$worker"/
