[![N|Solid](https://inside.epfl.ch/corp-id/wp-content/uploads/2019/05/EPFL_Logo_Digital_RGB_PROD-300x130.png)](https://nodesource.com/products/nsolid)
# EPFL Machine Learning - Higgs Challenge 2019

# About the Project
Using real CERN partcile accelerator data, we implemented a machine learning model able to detect the collision decay signature of a Higgs particle, in order to recreate the process that led to the discovery of the Higgs Boson.

# Folder Organisation
In this section we explain how our project folder is organised and where to find the needed files.

### Data Folder
**Please import the train.csv and the test.csv in this folder.**

All the CERN particle accelerator data as well as our predictions are found in the **/data** folder. It consists of:

1. **sample-submission.csv**: file containing our predictions when the run.py code is executed
2. **train.csv** : train set data (imported by the user)
3. **test.csv**: test set data (imported by the user)

### Scripts Folder
All our project implementation can be found within the **/scripts** folder.

**Python Executables .py**
1. **run**: gives us our best accuracy score on Aicrowd
2. **implementations**: contains all the required functions
3. **costs**: loss and accuracy functions
4. **cross_validation**: all cross validation methods that we used in our process
5. **data_preprocessing** : function that help us preprocess the data

**Jupyter Notebooks**
1. **project_1**: runs all different cross-validations for all methods, to obtain an accuracy for each method
2. **datapreprocessing_plot**: creates all graphs used for the preprocessing part of the report

# Running the code
In this section we explain how to run the code that provided our best submission on the Aicrowd EPFL Higgs Challenge.

1. Download our .zip project folder and exctract it
2. Open your terminal and from the root of the repository execute the following command
```sh
python3 scripts/run.py
```
3. After execution, the obtained predictions are available in the **sample-submission.csv** file
4. Upload this file on the Aicrowd Higgs Challenge





