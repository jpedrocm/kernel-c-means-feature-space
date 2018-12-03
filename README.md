# Project 1

![python](https://img.shields.io/badge/python-3.7-blue.svg)
![status](https://img.shields.io/badge/status-complete-brightgreen.svg)
![license](https://img.shields.io/badge/license-MIT-green.svg)

This is the first project for the Machine Learning class.

## Description

The goal of this project is to implement the KCM-F-GH algorithm described in Section 3.2 of the article ["Gaussian kernel c-means hard clustering algorithms with automated computation of the width hyper-parameters"](https://www.sciencedirect.com/science/article/abs/pii/S0031320318300712) published on Pattern Recognition in July of 2018. The experiments were run on the Image Segmentation test set from the UCI Machine Learning Repository.

## Getting Started

### Requirements

* [Python](https://www.python.org/) >= 3.7.0
* [NumPy](http://www.numpy.org/) >= 1.15.4
* [pandas](https://pandas.pydata.org/) >= 0.23.4
* [scikit-learn](http://scikit-learn.org/stable/) >= 0.20.0


### Installing

* Clone this repository into your machine
* Download and install all the requirements listed above in the given order
* Download the test set from the [Image Segmentation archive](http://archive.ics.uci.edu/ml/machine-learning-databases/image)
* Place the CSV's in the data/ folder
* Remove the CSV's first three lines
* Add a new "CLASS" column as the first header attribute

### Reproducing

* Edit the ConfigHelper attributes which are said to be configurable
* Run the experiments
```
python main.py
```

## Project Structure

    .
    ├── src                          # Source code files
    |   ├── main.py
    |   ├── kcm_fgh.py
    |   ├── config_helper.py
    |   ├── io_helper.py 
    |   └── metrics_helper.py
    ├── data                         # Dataset file
    ├── results                      # Experiment results
    ├── LICENSE.md
    └── README.md

## Author

* [jpedrocm](https://github.com/jpedrocm)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
