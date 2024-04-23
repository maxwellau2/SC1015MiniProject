# Welcome to the Air Quality Project

## About
This Mini-Project aims to analyse the PRSA dataset on Multi-State Air Quality in Beijing in an effort to build a solution to manage symptoms of poor air quality. Below is a flowchart of the reccomended path to look at our project.

<img src="img/flow_of_events.drawio.png"></img>

In order to make the notebook easier to follow, we have
1. **Abstracted data imputation** codes into a **datacleaning.py**
2. **Abstracted data transformation** functions into **datatransformations.py**
3. Abstracted the cycle shown above (Feature Engineering, Data Preparation, Model Training, Model Evaluation) into a single notebook **showing only evalutation steps** (models will be trained seperately to reduce code redundancy in the notebook).

**All abstractions are available to read in ./scripts**

This is in an effort to reduce the size of the notebook to make it easier to follow **(No one wants to read a monolith that is difficult to read, do they?)**. We wil add the general ideas of what each function does as markdown cells.

## Problem statement:


## Feature Engineering Techniques
### Dealing with skew
1. Logarithmic Tranformation
2. Square Root Transformation
3. Cox-Box Transformation
4. Yeo-Johnson Transformation

### Feature Generation
1. Orthogonal Projection
2. Time Series Data Creation

## Models used
### Classical Machine Learning Models (Regression Problems)
1. Random Forest Regressor
2. Adaboost Regressor
3. Linear Regression
4. **Neural Network Regression(MLP Regressor)**
   
### Time Series Models Used (Sequential Regression Problem)
1. LSTM Networks
2. **GRU Networks**
