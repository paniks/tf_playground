### Python tensorflow playground

##### Requirements (for Linux Ubuntu 18.04LTS with Python3.6.7)

Update system
```bash
sudo apt-get update
sudo apt-get upgrade
```

Prepare Python 3.5 env
```bash
sudo apt-get install python3.6
sudo pip3 install virtualenv

(it's my favorite way to manage python's virtualenvs, it's not obligatory or something)
mkdir ~/venvs
cd ~/venvs/

virtualenv django-blog -p $(which python3)
```

Activate virtualenv and install requirements
```bash
(in project dir)

source path/to/venv/bin/activate
pip install -r requiments.txt
```

There is possibility to install requirements manually, then you can add parallelisation to HDF5 files handling or GPU support for Tensorflow.  
Check:   
http://docs.h5py.org/en/stable/mpi.html  
https://www.tensorflow.org/install/