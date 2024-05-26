# Multi-task Learning Model for House Price Prediction and Classification
## Table of Contents
- [Introduction](#Introduction)
- [Problem Statement](#problem-statement)
- [Hypothesis](#hypothesis)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Univariate Analysis: House Price Distribution](#univariate-analysis-house-price-distribution)
  - [Univariate Analysis: House Category Distribution](#univariate-analysis-house-category-distribution)
  - [Bivariate Analysis: Some Key Numeric Features Against House Price](#bivariate-analysis-some-key-numeric-features-against-house-price)
  - [Bivariate Analysis: Some Key Categorical Features Against House Price](#bivariate-analysis-some-key-categorical-features-against-house-price)
  - [Multivariate Analysis: Correlation Matrix](#multivariate-analysis-correlation-matrix)
  - [Nullity](#nullity)
    - [Nullity Correlation](#nullity-correlation)
    - [Nullity Matrix](#nullity-matrix)
- [Model Architecture](#model-architecture)
- [Performance on Holdout Set](#performance-on-holdout-set)
## Introduction
In this project,I built and evaluated a multi-task learning model that simultaneously predicts house prices, a regression task, and categorizes houses into predefined categories, a classification task. By leveraging multi-task learning and utilizing advanced features of PyTorch Lightning, I efficiently managed and optimized the learning process for these intertwined tasks.

## Problem Statement
The challenge of this project is two-fold: to predict the sale price of houses and to classify them into categories based on age, building type, and renovation status. The goal is to develop a predictive model that can handle multiple output variables and exploit commonalities and differences between tasks to improve prediction accuracy and model efficiency.

## Hypothesis
The integration of house characteristics, such as building type and renovation history, into a unified model can significantly enhance the prediction performance over separate models handling each task independently. This hypothesis is based on the premise that certain features will have shared influence across both regression and classification tasks, leading to more robust and generalizable predictions.

## Dataset
The dataset utilized in this analysis is the ["House Prices - Advanced Regression Techniques"](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) from Kaggle. In addition to the original features, this project introduces a new categorical variable, 'House Category', derived from the 'House Style', 'Bldg Type', 'Year Built', and 'Year Remod/Add' features. The categorization is as follows:

- Houses renovated or built within the last 20 years are categorized as 'Modern'.
- Houses renovated or built within the last 50 years are classified as 'Contemporary'.
- Houses renovated or built within the last 100 years fall under 'Vintage'.
- Older houses are labeled as 'Historic'.

Additionally, houses are categorized based on building type and style into 'Family Home', 'Townhouse', and 'Multi-Family or Duplex', among others. The combination of age and type categories results in the comprehensive 'House Category' for each property. This enriched dataset provides a nuanced framework for the multi-task learning neural network to operate within.

## Exploratory Data Analysis

### Univariate Analysis: House Price Distribution 
![House Price Distribution](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/5b9d8f4f805194cb869fb55827aa9d6428f318d6/Plots/House%20price.png)
#### Insights from the SalePrice Distribution

- The distribution of SalePrice is right-skewed.
- Most properties are concentrated in the price range of \$100,000 to \$300,000.
- The peak (mode) of the distribution occurs between \$150,000 and \$200,000.
- There are fewer properties priced above \$300,000, indicating a tapering off of higher-priced homes.
- The presence of a long tail towards higher prices suggests a small number of high-value properties.


### Univariate Analysis: House Category Distribution
![House Category Distribution](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/39bed33ccc21bd6ea500bc0d503eca5308a180c7/Plots/House%20cat.png)
#### Insights from House Category Distribution

- **Contemporary Family Home** and **Vintage Family Home** are the most common house categories, each with over 400 occurrences.
- **Modern Family Home** follows closely with around 350 occurrences.
- **Contemporary Townhouse** and **Modern Townhouse** have a moderate presence with approximately 100 to 150 occurrences each.
- **Vintage Multi-Family or Duplex** and **Contemporary Multi-Family or Duplex** are less common, each with fewer than 50 occurrences.
- **Modern Multi-Family or Duplex** and **Vintage Townhouse** are the least common categories, with very few occurrences.

This distribution indicates a predominance of family homes, both contemporary and vintage, while multi-family and townhouse categories are less represented.

### Bivariate Analysis: Some Key Numeric Features Against House Price
![Bivariate Analysis](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/48d9521fae3a4ca9d36dc40a3d6f848155a9c8fc/Plots/Bivariate.png)
#### Insights from Bivariate Scatter Plots

1. **GrLivArea vs SalePrice**
   - There is a positive correlation between above-ground living area (GrLivArea) and SalePrice.
   - Higher GrLivArea generally corresponds to higher SalePrice.

2. **YearRemodAdd vs SalePrice**
   - Homes remodeled more recently tend to have higher SalePrices.
   - There is an increasing trend in SalePrice with more recent remodel years.

3. **GarageArea vs SalePrice**
   - A positive correlation exists between GarageArea and SalePrice.
   - Larger garage areas are associated with higher SalePrices.

4. **TotalBsmtSF vs SalePrice**
   - There is a strong positive correlation between Total Basement Area (TotalBsmtSF) and SalePrice.
   - Larger basements contribute to higher SalePrices.

5. **YearBuilt vs SalePrice**
   - Newer homes (more recent YearBuilt) tend to have higher SalePrices.
   - There is a noticeable increase in SalePrice for homes built after the 2000s.

6. **LotArea vs SalePrice**
   - While there is a positive trend, the correlation between LotArea and SalePrice is weaker compared to other features.
   - A few outliers with very large LotAreas show higher SalePrices, but the majority of data points are clustered with moderate LotArea.

### Bivariate Analysis: Some Key Categorical Features Against House Price
![Bivariate2](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/3e906e33cd7ea41a12b401c49a37b63b4a53df44/Plots/Bivariate2.png)
#### Insights from Additional Bivariate Box Plots

1. **HouseStyle vs SalePrice**
   - Two-story houses have the highest median SalePrice.
   - One-story houses also have a high median SalePrice, but with a wider range of prices.
   - 1.5 and 1-story unf houses have lower median SalePrices.

2. **SaleType vs SalePrice**
   - New houses (SaleType 'New') have the highest median SalePrice with a significant spread.
   - Houses sold through negotiation (SaleType 'WD') also have high median SalePrices.
   - Other sale types generally show lower median SalePrices.

3. **BldgType vs SalePrice**
   - Single-family detached houses (BldgType '1Fam') have the highest median SalePrice.
   - Townhouses (BldgType 'TwnhsE') and duplexes (BldgType 'Duplex') show lower median SalePrices.

4. **OverallQual vs SalePrice**
   - There is a strong positive correlation between overall quality (OverallQual) and SalePrice.
   - Higher quality ratings are associated with significantly higher SalePrices.
   - The median SalePrice increases steadily with the increase in OverallQual rating.

These box plots highlight how various categorical features like house style, sale type, building type, and overall quality influence the SalePrice, with higher quality and newer or detached houses generally commanding higher prices.

### Multivariate Analysis: Correlation Matrix
![Multivariate](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/070fdfc39f3aaee3815a64f27a415e0af0e846cd/Plots/heatmap.png)
#### Insights from Correlation Heatmap

- The heatmap reveals strong correlations between several features.
- **GarageCars** and **GarageArea** have a correlation of 0.88, indicating high multicollinearity.
- **TotalBsmtSF** and **1stFlrSF** are also highly correlated with a value of 0.82.
- **TotRmsAbvGrd** and **GrLivArea** show a strong correlation of 0.71.

#### Feature Dropping
To address multicollinearity, the following features were dropped:
- **GarageCars**
- **TotalBsmtSF**
- **TotRmsAbvGrd**



### Nullity 
#### Nullity Correlation
![Nullity](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/cfd7a9ecc0d4c7dbe7fc04c611ce9c5f7e2e6392/Plots/Screen%20Shot%202024-05-25%20at%207.53.20%20AM.png)
##### Insights from Nullity Correlation Heatmap

- The nullity correlation heatmap measures how strongly the presence or absence of one variable affects the presence of another.
- **GarageType**, **GarageYrBlt**, **GarageFinish**, **GarageCars**, **GarageArea**, **GarageQual**, and **GarageCond** exhibit perfect positive correlations (correlation value of 1.0) with each other, indicating that when one is present, all others are also present.
- **Electrical** has a weak positive correlation with **FireplaceQu** (0.2) and **GarageType** (0.2), suggesting a slight association between the presence of electrical features and these variables.
- Most other features have low or negligible correlations, indicating little to no association in their presence or absence.

##### Interpretation
The perfect positive correlations among garage-related features suggest that they often co-occur. This insight helped in handling missing data, as imputing one of these features provided information about the others. The weak correlations of other features imply that missingness in these variables is less likely to be influenced by the presence or absence of others.

This nullity correlation analysis is generally useful for understanding the patterns of missing data and informs the strategy for data imputation or feature engineering.

#### Nullity Matrix
![Nullity Matrix](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/924a5eabb26e18b340bddc1d969ff8aba497f47d/Plots/Screen%20Shot%202024-05-25%20at%207.53.31%20AM.png)
##### Insights from Nullity Matrix

- The nullity matrix provides a visual representation of missing data patterns across the dataset.
- Features such as **PoolQC**, **MiscFeature**, **Alley**, and **Fence** exhibit a high frequency of missing values.
- **LotFrontage** and **FireplaceQu** also have notable missing data, though less than the aforementioned features.
- The majority of the dataset has complete data, as indicated by the dense blue lines.
- The nullity matrix helps to quickly identify features with significant missing data, which are candidates for imputation or removal. Most of the missing data in this dataset is informative; therefore, they were replaced with the appropriate values accordingly

##### Interpretation
The presence of substantial missing values in certain features suggests the need for targeted data imputation strategies which, in this case, involved replacing nulls with the appropriate values such as "NoAlley", "NoFence", etc.

This visual aided in understanding the completeness of the dataset and informed decisions for handling missing data during the data cleaning process.


## Model Architecture

![Arch](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/04f93da5ca679df69eac1e89bccf2323e7723e03/Plots/Architecture.png)
The final architecture of the multi-task learning model is designed to simultaneously handle regression and classification tasks. Here are the key components:

**Shared Feature Layers:**
- A linear layer that expands the input features to 64 dimensions, followed by batch normalization and ReLU activation.
- Another linear layer that reduces the dimensions to 32, also followed by batch normalization and ReLU activation.

**Task-Specific Heads:**
- **Price Head:** A linear reduction to 16 features, ReLU activation, and a final linear output layer for price prediction (regression).
- **Category Head:** Similar structure to the Price Head but concludes with a linear layer for multi-class output (classification).

**Loss Functions and Metrics:**
- Mean Squared Error Loss for regression.
- Cross-Entropy Loss for classification.
- Performance metrics include RMSE and R2 Score for regression and accuracy, precision, recall, and F1 score for classification.

**Uncertainty Balancing:**
- Incorporation of task uncertainty parameters (`log sigma squared price` and `log sigma squared category`) to dynamically adjust the contribution of each task's loss.
- This approach balances the learning process, improves convergence, and enhances generalization across both tasks.

## Performance on Holdout Set

| Metric            | Value     |
|-------------------|-----------|
| Test Accuracy     | 0.76      |
| Test Cat Loss     | 0.90      |
| Test F1 Score     | 0.71      |
| Test Loss         | 1.72      |
| Test Precision    | 0.72     |
| Test Price RMSE   | 53735.53  |
| Test R2 Score     | 0.55      |
| Test Recall       | 0.72      |

### Note:
- **RMSE (Root Mean Squared Error)** is used instead of MSE as it is more intuitive and penalizes large errors more effectively. MAE could have been used but does not penalize large errors.


For more details, please refer to the [project report file](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/476991e53d2444001b4f6f9385e1e7f9a708202e/Report.pdf).
