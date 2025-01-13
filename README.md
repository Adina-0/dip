# Pollen Species Recognition: Digital Image Processing at HUST

## Authors:
Cù Minh Anh & Adina Waidhas

## Introduction
The project is designed to classify pollen species from bright-field microscopic images and evaluate
different descriptors (features) for the pollen grains.
Though in general, images of other kind of objects or acquisition methods might be able
to be processed with the pipeline as well, this hasn't been tested yet.
It follows a pipeline of image preprocessing, feature extraction (texture, geometry and color) and classification.
The classification is done in two separate ways: Random Forest and Fully Connected Neural Network.

## Data Requirements
All the data should be stored in one folder, the path can be specified in the main method ('Data' will be used by default).
Each pollen taxa (or in general, each object class) should have a folder named by the species (class) name,
containing all the images of that class. The images should be in the format of '.jpg' or '.png' and if
geometrical features are to be extracted, they should all share the same scale (e.g. 1 pixel = 1 mm). The size of
the images can be variable across the dataset, and images don't have to be square either.

## Installation
The project is written in Python 3.11. The packages required can be installed by running the following command:
```bash
pip install -r requirements.txt
```

## Usage
To start the pipeline, run the 'main.py' script after adapting the following parameters.
The user can specify the path to the data folder (_data_path_),
the type(s) of features (texture, geometric, color) to be extracted and the classification method(s)
to be used as boolean values. The variable _output_identifier_ specifies the prefix of the output files
generated. Depending on the size of the dataset and the number of features included, the process can take a while.
The output will be saved in the 'Results' folder.

## Results
The trained Random Forest model and its metadata are saved in the 'Results' folder as '.joblib' files.
They can be loaded and used for further classification tasks (as in the 'classify.py' script) provided.
Classification summaries are printed to the console. Just as for the plots generated and shown on the screen,
the user can save them manually.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
