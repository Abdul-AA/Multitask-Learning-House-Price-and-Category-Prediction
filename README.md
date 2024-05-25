# Multi-task Learning Model for House Price Prediction and Classification

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
## Insights from the SalePrice Distribution

- The distribution of SalePrice is right-skewed.
- Most properties are concentrated in the price range of \$100,000 to \$300,000.
- The peak (mode) of the distribution occurs between \$150,000 and \$200,000.
- There are fewer properties priced above \$300,000, indicating a tapering off of higher-priced homes.
- The presence of a long tail towards higher prices suggests a small number of high-value properties.


### Univariate Analysis: House Category Distribution
![House Category Distribution](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/39bed33ccc21bd6ea500bc0d503eca5308a180c7/Plots/House%20cat.png)

### Bivariate Analysis:Some Key Numeric Features Against House Price
![Bivariate Analysis](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/48d9521fae3a4ca9d36dc40a3d6f848155a9c8fc/Plots/Bivariate.png)
### Bivariate Analysis:Some Key Categorical Features Against House Price
![Bivariate2](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/3e906e33cd7ea41a12b401c49a37b63b4a53df44/Plots/Bivariate2.png)
### Multivariate Analysis: Correlation Matrix
![Multivariate](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/070fdfc39f3aaee3815a64f27a415e0af0e846cd/Plots/heatmap.png)


### Nullity 
#### Nullity Correlation
![Nullity](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/cfd7a9ecc0d4c7dbe7fc04c611ce9c5f7e2e6392/Plots/Screen%20Shot%202024-05-25%20at%207.53.20%20AM.png)
#### Nullity Matrix
![Nullity Matrix](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/924a5eabb26e18b340bddc1d969ff8aba497f47d/Plots/Screen%20Shot%202024-05-25%20at%207.53.31%20AM.png)

## Model Architecture

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
