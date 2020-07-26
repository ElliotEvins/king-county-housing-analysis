# King County Housing Data Analysis

# Introduction
## Executive Summary
This report provides an analysis and evaluation of the current and prospective housing market in King County, Washington. Methods include data and feature analysis, feature engineering, and linear/polynomial regression analysis. Other calculations involve model assessment and cross validation. All calculations can be found in the associated notebooks. Results of data analyzed produce applicable models in evaluating the sale price of a house in King County based on its identifying factors. 

The resulting models can be compounded in order to gain a menial understanding as to what an acceptable or expected selling price for a house could be based on its location and features. A more accurate model with a smaller margin of error would require further investigation but this analysis provides a template based on the available dataset.

Recommendations discussed include:
Price breakdown by area of King County
10 most cost-related zip codes

The report also investigates the fact that the analysis conducted has limitations. Some of the limitations include:
- Monetary figures are not nominalized, giving extra weight to more recent films. 
- Projections derived from regression analysis bare statistical shortcomings that inhibit suggestions from providing use outside their given ranges.
- Further analysis could utilize additional resources in order to identify additional content development strategies.

## Technologies 
This project was created using the following languages and libraries. An environment with the correct versions of the following libraries will allow re-production and improvement on this project.
- Python version: 3.6.9
- Matplotlib version: 3.0.3
- Seaborn version: 0.9.0
- Pandas version: 0.24.2
- Numpy version: 1.16.2
- Statsmodels 0.9.0
-Scikit Learn 0.4.0
 

# ETL - Blind Regression
This project begins with the provided dataset that encapsulates sales of houses in King County, Washington between 2014 and 2015. The dataset contains 21,597 unique non-null values in the largest variable and missing values in the `waterfront` and `yr_renovated` category. There are no duplicate rows but we find duplicate `id` values that represent houses that were sold at different times. After an initial investigation into these houses with multiple sells, they are included in the dataset as unique values that can contribute information to the business case. 

A quick search finds that the provided dataset is incomplete while the original provided by Kaggle has full values of the missing columns. Merging these two tables on the duplicate `id` and `date` values makes the decision of what to do with Nan values very easy. Once the table is full, we are able to adjust the datatypes and remove irrelevant values. 

To build a foundation of this project, we’ll take this cleaned dataset and plug it into a model against the sale price to see what kind of relationship we’re dealing with. Though the assumptions of linearity is not met by this model, an adjusted R-squared value of 0.696 and mostly statistically significant P-values gives us hope that we’ll be able to explain sales price through regression analysis. 

![title](/misc/scatter-matrix.png) - BASELINE MODEL GRAPH

## EDA and Model-1 - 

![title](/misc/scatter-matrix.png)
