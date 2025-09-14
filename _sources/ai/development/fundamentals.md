# Fundamentals
Fundamental concepts important to Machine learning

## Error
- **Total Prediction Error**/**Expected Generalization Error** - error when applying a trained model to predict on unseen data. Can be broken down into 3 main components. 
    - **Bias Error** - error from erroneous assumptions in the algorithm (leads to [underfitting](https://en.wikipedia.org/wiki/Overfitting#Underfitting)). 
    - **Variance Error** - error from sensitivity to small fluctuations in training set (leads to [overfitting](https://en.wikipedia.org/wiki/Overfitting)). Fit the model too much and you'll end up fitting to the noise too.
    - **Irreducible Error** - error inherent in any dataset or process that cannot be reduced regardless of model complexity or amount of data. This is the lower bound on total prediction error.
- The [Bias-Variance Tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) is an inescapable tradeoff for supervised learning.
    - you want your model to capture regularities in the training data and also generalize well to unseen data.
    - **High-Bias** Models are simpler, can't capture as much detail, so they won't fit to noise, but they may miss patterns in the training data, resulting in underfitting.
    - **High-Variance** Models can represent the training data set well but risk overfitting to noise or training data that isn't representative of the larger population.