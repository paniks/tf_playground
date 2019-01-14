## Tensorflow playground

It's 'part' of
[cardiorama](https://github.com/zamkot/cardiorama) which I'm co-coordinator. Cardiorama is about ECG analysis so examples in this repo are based on heart classes classification. This repo is kind of shortcut of using tensorflow lib in both C++ and Python for ones responsible for heart classes classification in out student project.

A descriptor class (not included to the repository) extracted some simple morphological features for each heartbeats in ECG signal. It also matched proper labels.
ECG signals come from MIT-BIH db by Physionet.

Features for heartbeats and their labels are kept in HDF5 files. This approach allows the use of features in both C ++ and Python without additional conversions, because HDF5 files have API for both languages.

##### get repo

```bash
cd directory
git clone https://github.com/paniks/blogapp.git
```

##### variations
[Python](https://github.com/paniks/tf_playground/python)  
[c++](https://github.com/paniks/tf_playground/cpp)