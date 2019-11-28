![Eaters Collective on Unsplash](https://images.unsplash.com/photo-1472393365320-db77a5abbecc?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1500&q=80)

# Predict Business Success & Failure
In this repository we use the Yelp Dataset to answer the following question:<br>
<b> Can we predict if a business is open or closed? What are the main indicators of viability? </b> <br>
We are able to predict our binary target with an AUC of 0.75 and provide an individual explanation for each individual business.

## Getting Started
All code is written in Python 3.x, it is preferable to open the files using Jupyter Notebooks.
All package dependecies are listed in the requirements.txt file.

To obtain the yelp dataset, follow this link:<br>
https://www.yelp.com/dataset/challenge

We use the following files:
* Businesses: (business.json) This dataset contains business relevant data including location data, restaurant attributes, and cuisine categories.
* Reviews: (Review.json) This dataset contains full review text data including the user_id that wrote the review, and the business_id the review is written for. 
* Checkin: (Checkin.json) This dataset contains the check ins for businesses where available.


For a detailed walkthrough, please check out the medium post:
https://medium.com/@nadimkawwa/how-to-predict-business-success-failure-3c11b9c3a04c

## Project Structure
The repository is ordered as specified in the tree below.
Note that notebooks are numbered in a suggested order of execution. 

    .
    ├── data                    # data artifacts
    ├── models                  # saved models
    ├── plots                   # generated plots
    ├── predictions             # contains supervisedl earning notebooks
    │   ├── gbtree              # gradient boosted trees
    │   ├── keras               # keras
    │   ├── logreg              # logistic regression
    │   ├── rftree              # random forest
    │   └── xgb                 # xgboost
    └── wrangle                 # data wrangling + feature extracton

## Results & Model Selection
After feature wrangling, the selection of a winning model is highlighted in green below:
![model performance](https://github.com/NadimKawwa/nadim_sharpest_minds/blob/predictions_reordering/plots/summary_performance.png)

We can also extract the main features for selected top performers:
![feature importance](https://github.com/NadimKawwa/nadim_sharpest_minds/blob/predictions_reordering/plots/summary_features.png)

## Model Interpretation 
We are also interested in interpreting outputs on a per user basis. We therefore show the main SHAP values in the plots below. The x-axis shows the average magnitude change in the model output when a feature is hidden from the model. Given that hiding a feature changes depending on what other features are also hidden, Shapley values are used to enforce consistency and accuracy.
![mean absolute SHAP values](https://github.com/NadimKawwa/nadim_sharpest_minds/blob/master/plots/shap_summary_plot.png)

A density scatter plot of SHAP values for each feature to identify how much impact each feature has on the model output for each observation in the dataset. The summary plot combines feature importance with feature effects. Each point on the summary plot is a Shapley value for a feature and an instance. The position on the y-axis is determined by the feature and on the x-axis by the Shapley value. The color represents the value of the feature from low to high.
![shap density scatter plot](https://github.com/NadimKawwa/nadim_sharpest_minds/blob/master/plots/xgb_shap_density.png)

  
## Acknowledgments

* Aditya Subramanian: https://github.com/adityasubramanian
* SharpestMinds: https://www.sharpestminds.com/
* https://github.com/Yelp/dataset-examples
* Christoph Molnar: https://christophm.github.io/interpretable-ml-book/
* Scott Lundberg: https://scottlundberg.com/
* Abhay Pawar: https://github.com/abhayspawar/featexp
* Michail Alifierakis: https://michailalifierakis.com/
* IRS data: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-2016-zip-code-data-soi
* Statistics Canada: https://www12.statcan.gc.ca/census-recensement/2016/dp-pd/hlt-fst/inc-rev/Table.cfm?Lang=Eng&T=102&PR=0&D1=1&RPP=25&SR=1&S=108&O=D
