#include <iostream>
#include "H5Cpp.h"

#define PRINT(x) std::cout<<x<<std::endl;

const H5std_string FILE_NAME("../../data/features.h5");
const H5std_string FEATURES_DATASET("features");
const H5std_string LABELS_DATASET("labels");


int main() {

    H5::H5File file(FILE_NAME, H5F_ACC_RDONLY);

    // --------------- load features ---------------------

    H5::DataSet featuresDataSet = file.openDataSet(FEATURES_DATASET);
    H5::DataType featuresDType = featuresDataSet.getDataType();
    H5::DataSpace dataSpace = featuresDataSet.getSpace();

    auto dims = dataSpace.getSimpleExtentNdims();
    hsize_t dimsOut[dims];
    dataSpace.getSimpleExtentDims(dimsOut, nullptr);

    hsize_t dimsMatrix[dims];
    for(int i = 0; i<dims;i++){
        dimsMatrix[i] = dimsOut[i];
    }

    H5::DataSpace memSpace( dims, dimsMatrix);

    float features[(int)dimsOut[0]][(int)dimsOut[1]];
    featuresDataSet.read(features, featuresDType, memSpace, dataSpace);

    PRINT("FEATURES LOADED");

    // --------------- load labels ---------------------

    H5::DataSet labelsDataset = file.openDataSet(LABELS_DATASET);
    H5::DataType labelsDType = labelsDataset.getDataType();

    char *labels[static_cast<int>(dimsOut[0])];
    labelsDataset.read((void*)labels,labelsDType);

    std::cout << "Hello, World!" << std::endl;

    PRINT("LABELS LOADED");

    return 0;
}