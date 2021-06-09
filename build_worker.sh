worker=$1

mkdir "$worker"
mkdir "$worker"/codec
mkdir "$worker"/dataset
mkdir -p "$worker"/executor/psgd

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
