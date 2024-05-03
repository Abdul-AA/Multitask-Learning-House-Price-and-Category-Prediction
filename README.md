# Multi-task Learning Model for House Price Prediction and Classification

In the dynamic and complex real estate market, accurate prediction of house prices and classification of house types are crucial for both market analysis and investment decisions. This project aims to build and evaluate a multi-task learning model that simultaneously predicts house prices, a regression task, and categorizes houses into predefined categories, a classification task. By leveraging multi-task learning models and utilizing advanced features of PyTorch Lightning, this project seeks to efficiently manage and optimize the learning process for these intertwined tasks.

## Problem Statement
The challenge of this project is two-fold: to predict the sale price of houses and to classify them into categories based on age, building type, and renovation status. The goal is to develop a predictive model that can handle multiple output variables and exploit commonalities and differences between tasks to improve prediction accuracy and model efficiency.

## Hypothesis
We hypothesize that the integration of house characteristics, such as building type and renovation history, into a unified model can significantly enhance the prediction performance over separate models handling each task independently. This hypothesis is based on the premise that certain features will have shared influence across both regression and classification tasks, leading to more robust and generalizable predictions.

## Dataset
The dataset utilized in this analysis is the ["House Prices - Advanced Regression Techniques"](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) from Kaggle. In addition to the original features, this project introduces a new categorical variable, 'House Category', derived from the 'House Style', 'Bldg Type', 'Year Built', and 'Year Remod/Add' features. The categorization is as follows:

- Houses renovated or built within the last 20 years are categorized as 'Modern'.
- Houses renovated or built within the last 50 years are classified as 'Contemporary'.
- Houses renovated or built within the last 100 years fall under 'Vintage'.
- Older houses are labeled as 'Historic'.

Additionally, houses are categorized based on building type and style into 'Family Home', 'Townhouse', and 'Multi-Family or Duplex', among others. The combination of age and type categories results in the comprehensive 'House Category' for each property. This enriched dataset provides a nuanced framework for our multi-task learning model to operate within.

For more details, please refer to the [project report file](https://github.com/Abdul-AA/Multitask-Learning-House-Price-and-Category-Prediction/blob/476991e53d2444001b4f6f9385e1e7f9a708202e/Report.pdf).
