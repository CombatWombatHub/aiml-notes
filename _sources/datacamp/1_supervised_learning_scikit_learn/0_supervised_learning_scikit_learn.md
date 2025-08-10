# Supervised Learning with scikit-learn
- [Supervised Learning with scikit-learn](https://app.datacamp.com/learn/courses/supervised-learning-with-scikit-learn)
- first course on the track "[Machine Learning Scientist in Python](https://app.datacamp.com/learn/career-tracks/machine-learning-scientist-with-python)"
- **Supervised Learning** - the values to be predicted are already known, goal is to predict values of previously unseen data

## Supervised Learning Basics

### Types
- **Classification** - predict the label or category of an observations (is a transaction fraudulent or not)
- **Regression** - predict continuous variables (cost of house based on size, bedrooms,...)

### Terminology
- **features** - independent variables, predictor variables, variables being input
- **target variable** - dependent variable, response variable, variable being predicted

### Data Prerequisites
- data must not have missing values
- must be numeric
- usually we store in Pandas DataFrames or NumPy arrays
- do Exploratory Data Analysis to check it out first

### scikit-learn Syntax
- [scikit-learn](https://scikit-learn.org/stable/)
- that page actually has good way to select categories like classification, regression, clustering, dimensionality reduction, model selection, preprocessing