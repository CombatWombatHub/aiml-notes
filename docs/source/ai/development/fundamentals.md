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

## Tokenization
- You'd think this was isolated to [Natural Language Processing](../categories/nlp.md), but no.
- if you want to use an [Unsupervised](../categories/ml.md) model to cluster documents into categories, you need to tokenize the words there as well
- tokenization is breaking strings into chunks (tokens). They're often words, but sometimes not - you  might include the period after a word to capture the fact that that word means something different because it comes at the end of a sentence, or you might grab the `ing` from the end of a word indicating it's an ongoing action.

### TF-IDF (Term Frequency - Inverse Document Frequency) Matrix
- **TF-IDF** is a weighted measure of how important a word is to a document in a collection. It can be used to make a matrix. 
- It multiplies:
    - **TF (Term Frequency)** - how often a specific word appears in a single document, filtering out words that are probably just one-offs or from a bibliography or something
    - **IDF (Inverse Document Frequency)** - how exclusive a word is to a single document (words that appear a lot in all documents will have a low **IDF**, filtering out common words like `"the"`)
- so basically a word that appears many times in the current document (high **TF**) but doesn't appear much in *most* of the *other* documents (high **IDF**) will have a really high **TF-IDF** score
    - each unique word in the document could be assigned a **TF-IDF** value, 
    - you can a matrix out of all the **TF-IDF**'s for each document
    - then you could cluster from the matrix with K-Means or something
    - you'd assume that documents with similar high-**TF-IDF** words are on similar topics