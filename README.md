# Welcome to the Air Quality Project

## About
This Mini-Project aims to analyse the PRSA dataset on Multi-State Air Quality in Beijing in an effort to build a solution to manage symptoms of poor air quality. The data originates from the Beijing Municipal Environmental Monitoring Center, with meteorological information at each station sourced from the closest weather station managed by the China Meteorological Administration. The observations span from March 1st, 2013, to February 28th, 2017.<div>Data set available <a href="https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data">here</a></div> 

Below is a flowchart of the reccomended path to look at our project.


<img src="img/flow_of_events.drawio.png"></img>

In order to make the notebook easier to follow, we have
1. **Abstracted data imputation** codes into a **<a href="./scripts/datacleaningutils.py">datacleaningutils.py</a>**
2. **Abstracted data transformation** functions into **<a href="./scripts/datatransformations.py">datatransformations.py</a>**
3. **Abstracted Regression Modelling Processes** into **<a href="./scripts/classical_models.py">classical_models.py</a>**
4. **Abstracted Time Series Modelling Processes** into **<a href="./scripts/timeseries_trainer.py">timeseries_trainer.py</a>**
5. Abstracted the cycle shown above (Data Preparation, Model Training, Model Evaluation) into a single notebook **showing only evalutation steps** (models will be trained seperately to reduce code redundancy in the notebook).

This is in an effort to reduce the size of the notebook to make it easier to follow **(No one wants to read a monolith that is difficult to read, do they?)**. We will add the general ideas of what each function does as markdown cells.

### Reading Order
1. <a href="./businessproposition.md">Business Proposition</a>
2. <a href="./data_imputation.ipynb">Data Imputation</a>
3. <a href="./EDA.ipynb">Exploratory Data Analysis</a>
4. <a href="./feature_engineering.ipynb">Feature Engineering</a>
5. <a href="./model_eval.ipynb">Model Evaluation (Too large to view, please download to see)</a>
6. <a href="./productdesign.md">Product Design</a>

## Dataset

This dataset contains hourly records of air pollutant levels from 12 government-monitored air quality stations. The data originates from the Beijing Municipal Environmental Monitoring Center, with meteorological information at each station sourced from the closest weather station managed by the China Meteorological Administration. The observations span from March 1st, 2013, to February 28th, 2017, with any unavailable data marked as "NA."


## Problem statement/Business Proposition:
The American Heart Association warns that exposure to fine particulate matter (PM2.5) can significantly increase the risk of cardiovascular diseases and mortality, especially with prolonged exposure. In response, we aim to develop a data-driven system, a Simple Reflex Agent with State (SRAS), to mitigate PM2.5 exposure indoors. This leads us to the questions regarding this dataset.

1. Can we predict the next time step's PM2.5?
2. If the PM2.5 sensor is down, can we estimate current PM2.5?

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

## Solution
### Create idea of how to implement models onto a Simple Reflex Agent with State.
Our solution entails the development of a Simple Reflex Agent with State (SRAS) to tackle the health hazards linked to indoor exposure to fine particulate matter (PM2.5), drawing on insights from the American Heart Association's alerts.
Implemented using a Raspberry Pi for cost-effectiveness and ease of deployment, our SRAS continually monitors environmental conditions via sensors and timers, utilizing predictive algorithms to anticipate future PM2.5 levels.

## Contributors
- **Muthukrishnan Sushruth** (EDA)
- **Rana Khushi** (Data Imputation, Data Cleaning, Video Editor)
- **Au Ze Hong, Maxwell** (EDA, Data Imputation, Feature Engineering, Regression Modelling, Time Series Modelling, File Organizer, Package Developer, Product Designer)

## What did we learn?
- Handling skewed data with transformations
- Principal Component Analysis
- Creative feature extraction (projection)
- Plotly graphing
- More machine learning models (Adaboost, Random Forest, Neural Network)
- Time Series Forecasting (GRU, LSTM)
- New metrics (MAPE)
- Researching articles (most of the stuff we did were from looking through articles, and interpreting the implementation)

## References

https://www.nature.com/articles/s41561-023-01157-8#:~:text=With%20emptier%20streets%20and%20quieter,and%2023%3A00%20globally4

https://airly.org/en/why-is-air-quality-worse-at-night/#:~:text=But%20at%20night%2C%20the%20ground,pollutants%20close%20to%20the%20surface

https://whnt.com/weather/why-was-it-so-windy-after-storms/

https://www.chinahighlights.com/beijing/weather.htm#:~:text=The%20rainy%20summer%20season%20is,moderate%20to%20heavy%20thundery%20showers.

https://aqli.epic.uchicago.edu/policy-impacts/china-national-air-quality-action-plan-2014/#:~:text=Across%20all%20urban%20areas%2C%20the,20%2C%2015%20percent%2C%20respectively.

https://www.business-standard.com/article/pti-stories/china-has-310-mln-registered-vehicles-385-mln-drivers-in-2017-118011501092_1.html

https://www.sciencedirect.com/science/article/abs/pii/S1352231023001231

https://www.waikatoregion.govt.nz/environment/air/weather-and-air/#:~:text=or%20no%20wind.-,Wind%20speed,in%20dry%20windy%20rural%20areas.

https://www.sciencedirect.com/science/article/abs/pii/S0048969723010501#:~:text=Through%20the%20combined%20effect%20of,autumn%20and%20winter%20in%20YRD.

https://aqicn.org/city/beijing/

https://blissair.com/what-is-pm-2-5.htm

https://textbooks.math.gatech.edu/ila/projections.html

https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data

https://builtin.com/data-science/step-step-explanation-principal-component-analysis

https://statisticaloddsandends.wordpress.com/2021/02/19/the-box-cox-and-yeo-johnson-transformations-for-continuous-variables/

https://medium.com/@kyawsawhtoon/log-transformation-purpose-and-interpretation-9444b4b049c9#:~:text=What%20is%20Log%20Transformation%3F,the%20purposes%20of%20statistical%20modeling.

https://quantifyinghealth.com/square-root-transformation/