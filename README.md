# King County Housing 

## Executive Summary
This report provides an analysis and evaluation of the current and prospective housing market in King County, Washington. Methods include data and feature analysis, feature engineering, and linear/polynomial regression analysis. Other calculations involve model assessment and cross validation. All calculations can be found in the associated notebooks. Results of data analyzed produce applicable models in evaluating the sale price of a house in King County based on its identifying factors. 

The resulting models can be compounded in order to gain a menial understanding as to what an acceptable or expected selling price for a house could be based on its location and features. A more accurate model with a smaller margin of error would require further investigation but this analysis provides a template based on the available dataset.

Recommendations discussed include:
Most and least expensive areas to purchase a home in King County
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

![title](/misc/baseline-model.png) - BASELINE MODEL GRAPH

## EDA and Model-1 - 

This model began with the import of our cleaned data and a quick look at values within the table. After looking at the descriptive values we plotted histograms for each variable : 

![title](/misc/hist-plot-1.png)

Looking at this list we can see that we have a `date` variable so we go ahead and engineer some initial variables by converting the datatype to datetime and creating a `month`, `age`,  and `day_of_year` variable. We also go ahead and create a `reno_age` variable based on the age of the last renovation.

### Categorical Variables

Visualizations help understand the data in some of the categorical variables `Bedrooms, Floors, Bathrooms / Bedrooms: 

![title](/misc/box-plot-1.png)

Doesn't look like there's a strong linear relationship here but maybe some of them are related with oneanother - can you say 3d visualizations? 

![title](/misc/3d-1.png)

Interesting but a little unnecessary at this point as we will want to asses these relationships later in multicolinear relationships within our model.  Let's look at other categoricals for relationship with price before we decide to dummy them out: 

![title](/misc/box-plot-2.png)

Based on this cursory look we’ll want to convert `['floors','waterfront','yr_renovated','sqft_basement']` to more binary variables that we’ll call `['multiple_stories','on_water','renovated','has_basement']`.

### Continuous Variables 

Just by filtering values we filter continuous variables into `['id', 'date', 'price', 'sqft_living', 'sqft_lot', 'sqft_above', 'yr_built', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'age']` then plot histograms again: 

![title](/misc/hist-plot-2.png)

These don’t look good so we clean them up with a logarithmic transformation and drop the originals: 

![title](/misc/hist-plot-3.png)

### Multicollinearity Check

From the following matrix we determine that many of the variable s involving sqft show high multicollinearity so we decide to drop them and engineer some new features

![title](/misc/mm-matrix.png)

## Feature Engineering and Model Assessment

To utilize some of the features hidden within our current data set we use the months of the year to create `spring` and `summer` variables. Similarly we split each zipcode value into its own dummy variable. Since this initially widened the dataset too much, we ran a quick regression model and plucked those zipcodes with the highest positive coefficients to include in this model. Our top ten that are currently included are: 

`['zip_98039', 'zip_98004', 'zip_98112', 'zip_98040', 'zip_98102', 'zip_98199', 'zip_98109', 'zip_98105', 'zip_98119', 'zip_98115',]`

Once we build out a proper model we find an ### Adjusted R-Squared of .710 #### and all P-Values below Alpha. The residuals for this model however reveal some pretty discomforting errors from this model - making it unviable for the moment. 

![title](/misc/engineered-model.png)

## Model-2 -Logarithmic Transformation of the Target  
This approach to building a model will look deeper at each variable before incorporating it into a model and make new attempts at more accurate feature engineering. The priority for this model is to mitigate RMSE and fulfill the assumptions of linearity.
[Notebook Associated with this Section](https://github.com/ElliotEvins/king-county-housing-analysis/blob/master/Notebooks/3_model_2_log_target.ipynb)

We begin with the data and collection of variables from the last model and decide which factors to remove from the model moving forward. `id` and `date` useful to keep as an index but we don’t need it in our model. Similarly, though we’re transforming the target variable to train this model, we’ll need to keep `price` in the data set to verify our predictions. `[‘day_of_year’,’lat’,’long’,’zipcode’]` are all redundant in our model so we drop them as well. 

![title](/misc/log-hist.png)

With this change we have a slight bump in R-Squared value and much more normalized residuals. 

![title](/misc/qq-1.png)

From there we can engineer a new feature to see if we can strengthen the model. Looking at the distribution of sales with an overlap of price indicated by hue: 

![title](/misc/lat-long-dist.png)

A closer look at the priciest houses show that they are above a latitude of 47.5 Degrees and East of -122.2. With this distribution we can divide the area into 4 sections and introduce each section as a categorical variable in the next model. This model gives us an R-Squared value of 0.823 with immaculate P-Values. Running the same model against the testing data provides near identical output furthering the validity.

![title](/misc/engineered-model-2.png)

Further engineering is possible from here by utilizing outside data sources that can reflect larger macroeconomic sentiments that usually influence the housing market. The Bureau of Labor and Statistics CSI could help provide this model with an explanation of the business cycle or recessions that can influence prices. 

## Model-3 - Polynomial Regression
This approach to building a model will once again incorporate the deeper at look into each variable before incorporating it into a model. We will attempt to apply higher-order relationships and build a polynomial regression. We begin by building a simplified version of our last model by taking out the location oriented variables. Though this knocks down the statistical significance of the model, it will allow us to run higher order models without destroying our computers. 

Moving from higher features to lower ones we test at each level and find that 2 features yields the most relevant model in this case. After splitting the training and testing data we find that the model is valid though not quite as strong as our previously engineered linear regression model. A further avenue of exploration woud be to approach our previously engineered variables based on location and apply higher order regression through this 2 feature approach. 

![title](/misc/engineered-model-3.png)
# Recommendations
Whether as a real-estate professional, homeowner, or hopeful homeowner - consult these three approaches to assess an estimated selling price of a home in King County.
If seeking value for the house then stay away from the top 10 zipcodes identified in this study, which come with a premium.
The negative coefficient on renovations tells us that expensive makeovers might add as much value to the price of a home as they may cost
Similarly multiple stories and basements are not as significant as most other factors like sqft living space, condition, or even the season in which the home is on the market! 
# Conclusion
This broadly scoped analysis yielded actionable insights and opportunities for further investigation. The findings herein will help narrow the field of possibility in the real estate transaction process and increase the chances of sustainability and success in some of the largest purchases individuals will ever make. While the preceding investigations are based on significant bodies of data, the deliverables and conclusions reached are intended to act as guidelines and a starting point for future investigations. Multiple avenues of further exploration have been presented over the course of this investigation and we have aimed to clearly identify our thinking for each step of the research process. Moving forward our models, findings, and approaches - presented here as deliverables - should be revisited with updated data and compared with realized future sales. 
