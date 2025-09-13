# Machine Learning (ML)
[Machine Learning](https://www.geeksforgeeks.org/machine-learning/machine-learning/) is a branch of Artificial Intelligence that focuses on *models* and [algorithms](https://www.geeksforgeeks.org/machine-learning/machine-learning-algorithms/) that let computers learn from data and improve from previous experience without being explicitly programmed. There are many [types](https://www.geeksforgeeks.org/machine-learning/types-of-machine-learning/) of machine learning.
- [Supervised Learning](https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/) - Use labeled data
    - [Classification](https://www.geeksforgeeks.org/machine-learning/getting-started-with-classification/) - Predict *categorical* (discrete) values
    - [Regression](https://www.geeksforgeeks.org/machine-learning/regression-in-machine-learning/) - Predict continuous numerical values
    - Both - some models can be for both Classification and Regression
        - [Ensemble Learning](https://www.geeksforgeeks.org/machine-learning/a-comprehensive-guide-to-ensemble-learning/) - Combine multiple simple models into one better model.
- [Unsupervised Learning](https://www.geeksforgeeks.org/machine-learning/unsupervised-learning/) - Use unlabeled data
    - [Clustering](https://www.geeksforgeeks.org/machine-learning/clustering-in-machine-learning/) - Group data into clusters based on similarity
    - [Dimensionality Reduction](https://www.geeksforgeeks.org/machine-learning/dimensionality-reduction/) - Simplify datasets by reducing features while keeping important information (often used to select features for other models)
    - [Association Rule Mining](https://www.geeksforgeeks.org/machine-learning/association-rule/) - Discover rules where the presence of one item in a dataset indicates the probability of the presence of another
- [Reinforcement Learning](https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/) - Learn from rewards by interacting with environment via trial and error
- [Forecasting Models](https://www.kaggle.com/code/ryanholbrook/forecasting-with-machine-learning) - Use past data to predict future trends (often time series problems)
- [Semi-Supervised Learning](https://www.geeksforgeeks.org/machine-learning/ml-semi-supervised-learning/) - Use some labeled data with more unlabeled data
- [Self-Supervised Learning](https://www.geeksforgeeks.org/machine-learning/self-supervised-learning-ssl/) - Generates its own labels from unlabeled data

## Supervised Learning
[Supervised Learning](https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/) uses labeled data

### Classification
[Classification](https://www.geeksforgeeks.org/machine-learning/getting-started-with-classification/) - Predict *categorical* (discrete) values
- Linear Classifiers
    - [Logistic Regression](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/) - Draws a sigmoid curve, predicts 0 or 1 if above or below curve. Despite "Regression" being in the name, it's for Classification
    - [Single-Layer Perceptron](https://www.geeksforgeeks.org/python/single-layer-perceptron-in-tensorflow/) - a single layer with a single neuron? Why?
    - [SGD (Stochastic Gradient Descent) Classifier](https://www.geeksforgeeks.org/python/stochastic-gradient-descent-classifier/) - adjust model parameters in the direction of the loss function's greatest gradient
- Non-Linear Classifiers
    - [KNN (K-Nearest Neighbors)](https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/) ([Notebook](../../datacamp/1_supervised_learning_scikit_learn/1_classification.ipynb)) - simple, looks at closest data points (neighbors) to make predictions based on similarity
    - [Naive Bayes](https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/) ([Gaussian](https://www.geeksforgeeks.org/machine-learning/gaussian-naive-bayes/), [Multinomial](https://www.geeksforgeeks.org/machine-learning/multinomial-naive-bayes/), [Bernoulli](https://www.geeksforgeeks.org/machine-learning/bernoulli-naive-bayes/), [Complement](https://www.geeksforgeeks.org/machine-learning/complement-naive-bayes-cnb-algorithm/)) - predicts the category of a data point with probability

### Regression
[Regression](https://www.geeksforgeeks.org/machine-learning/regression-in-machine-learning/) predict continuous numerical values
- [Linear Regression](https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/) - fit a straight line to the data with [Least Squares Method](https://www.geeksforgeeks.org/maths/least-square-method/)
- [Multiple Linear Regression](https://www.geeksforgeeks.org/machine-learning/ml-multiple-linear-regression-using-python/) - Extends Linear Regression to use multiple input variables
- [Polynomial Regression](https://www.geeksforgeeks.org/machine-learning/python-implementation-of-polynomial-regression/) - a polynomial curve fit
- [Lasso Regression (L1 Regularization)](https://www.geeksforgeeks.org/machine-learning/ridge-regression-vs-lasso-regression/) - regularized versions of linear regression that help avoid overfitting by penalizing the *absolute value* of large coefficients
- [Ridge Regression (L2 Regularization)](https://www.geeksforgeeks.org/machine-learning/ridge-regression-vs-lasso-regression/) - regularized versions of linear regression that help avoid overfitting by penalizing the *square* of large coefficients

### Both
These models can be used for both Regression and Classification
- [SVM (Support Vector Machine)](https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/) - find a hyperplane that separates classes of data. The [SVR (Support Vector Regression)](https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/) uses it for regression tasks by finding the hyperplane that minimizes the residual sum of squares.
- [Multi-Layer Perceptron](https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/) - your classic neural network
- [Decision Trees](https://www.geeksforgeeks.org/machine-learning/decision-tree-algorithms/) ([Introduction](https://www.geeksforgeeks.org/machine-learning/decision-tree-introduction-example/)) ([Classification](https://www.geeksforgeeks.org/machine-learning/building-and-implementing-decision-tree-classifiers-with-scikit-learn-a-comprehensive-guide/)) ([Regression](https://www.geeksforgeeks.org/machine-learning/python-decision-tree-regression-using-sklearn/)) - hierarchical tree structure that works like a flow chart. splits data into branches based on feature values. Often used as building blocks for Ensemble methods. ([CART (Classification and Regression Trees)](https://www.geeksforgeeks.org/machine-learning/cart-classification-and-regression-tree-in-machine-learning/)) is based on ([ID3 (Iterative Dichotomiser 3)](https://www.geeksforgeeks.org/machine-learning/iterative-dichotomiser-3-id3-algorithm-from-scratch/)), and is a specific algorithm for building decision trees that can be used for both classification and regression.

#### Ensemble Learning
[Ensemble Learning](https://www.geeksforgeeks.org/machine-learning/a-comprehensive-guide-to-ensemble-learning/) combines multiple simple models into one better model
- [Bagging (Bootstrap Aggregating) Method](https://www.geeksforgeeks.org/machine-learning/What-is-Bagging-classifier/) - Train models independently on different subsets of the data, then combine their predictions
    - [Random Forest](https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/) ([Classification](https://www.geeksforgeeks.org/dsa/random-forest-classifier-using-scikit-learn/)) ([Regression](https://www.geeksforgeeks.org/machine-learning/random-forest-regression-in-python/)) ([Hyperparameter Tuning](https://www.geeksforgeeks.org/machine-learning/random-forest-hyperparameter-tuning-in-python/)) - create many decision trees, train each on random parts of data, combine results via voting (for classification) or averaging (for regression)
    - [Random Subspace Method](https://www.machinelearningmastery.com/random-subspace-ensemble-with-python/) - train on random subsets of input features to enhance diversity and improve generalization while reducing overfitting.
- [Boosting Method](https://www.geeksforgeeks.org/machine-learning/boosting-in-machine-learning-boosting-and-adaboost/) - Train models sequentially, each model focusing on errors of prior models, then do weighted combination of their predictions
    - [AdaBoost (Adaptive Boosting)](https://www.geeksforgeeks.org/machine-learning/implementing-the-adaboost-algorithm-from-scratch/) - for challenging examples, assign weights to data points, combine weak classifiers with weighted voting
    - [GBM (Gradient Boosting Machines)](https://www.geeksforgeeks.org/machine-learning/ml-gradient-boosting/) - sequentially build decision trees, each tree correcting errors of previous ones
    - [XGBoost (Extreme Gradient Boosting)](https://www.geeksforgeeks.org/machine-learning/xgboost/) - optimizes like regularization and parallel processing for robustness and efficiency
    - [CatBoost (Categorical Boosting)](https://www.geeksforgeeks.org/machine-learning/catboost-ml/) - handles categorical features natively without extensive preprocessing
- [Stacking (Stacked Generalization) Method](https://machinelearningmastery.com/implementing-stacking-scratch-python/) - train multiple different models (often different types), use predictions as inputs to final "meta-model"

## Unsupervised Learning
[Unsupervised Learning](https://www.geeksforgeeks.org/machine-learning/unsupervised-learning/) uses unlabeled data

### Clustering
[Clustering](https://www.geeksforgeeks.org/machine-learning/clustering-in-machine-learning/) groups data into clusters based on similarity

#### Centroid-Based Methods
Centroid-Based (Partitioning) Methods organize data around central prototypes (centroids), each cluster represented by the mean or medoid of its members
- [K-Means Clustering](https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/) ([Elbow Method to find optimal K](https://www.geeksforgeeks.org/machine-learning/elbow-method-for-optimal-value-of-k-in-kmeans/)) ([KMeans++ Clustering](https://www.geeksforgeeks.org/machine-learning/ml-k-means-algorithm/)) ([K-Mode Clustering](https://www.geeksforgeeks.org/machine-learning/k-mode-clustering-in-python/)) - groups data into K clusters based on how close the points are to each other. Iteratively assigns points to the nearest centroid, recalculating centroids after each addition.
- [K-Medoids Clustering](https://www.geeksforgeeks.org/machine-learning/ml-k-medoids-clustering-with-example/) - similar to K-means, but uses actual data points (medoids) as the centers, making more robust to outliers
- [FCM (Fuzzy C-Means Clustering)](https://www.geeksforgeeks.org/machine-learning/ml-fuzzy-clustering/) - similar to K-means but uses Fuzzy Clustering, allowing each data point to belong to multiple clusters with varying degrees of membership

#### Distribution-Based Methods
- [GMM (Gaussian Mixture Model)](https://www.geeksforgeeks.org/machine-learning/gaussian-mixture-model/) - fits data as a weighted mixture of Gaussian distributions and assigns data points based on likelihood
- [Expectation-Maximization Algorithm](https://www.geeksforgeeks.org/machine-learning/ml-expectation-maximization-algorithm/) - good at finding unknown parameters using Expectation Step (calculating expected values of missing/hidden variables) and Maximization Step (maximizing log-likelihood to see how well the model explains the data)
- [DPMMs (Dirichlet Process Mixture Models)](https://www.geeksforgeeks.org/machine-learning/dirichlet-process-mixture-models-dpmms/) - flexible clustering method that can automatically decide the number of clusters based on the data (you don't have to specify beforehand like K-Means)

#### Connectivity-Based Methods
Connectivity-Based Methods (Hierarchical Clustering) build nested groupings of data by evaluating connections between data points using tree-like structure
- [Hierarchical Clustering](https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering/) - create clusters by building a tree step-by-step, merging or splitting groups 
- [Agglomerative Clustering](https://www.geeksforgeeks.org/machine-learning/agglomerative-methods-in-machine-learning/) - (Bottom-up) start with each point as a cluster and iteratively merge the closest ones
- [Divisive Clustering](https://www.geeksforgeeks.org/artificial-intelligence/divisive-clustering/) - (Top-down) starts with one cluster and splits iteratively into smaller clusters
- [Spectral Clustering](https://www.geeksforgeeks.org/machine-learning/ml-spectral-clustering/) - groups data by analyzing connections between points using graphs
- [AP (Affinity Propagation)](https://www.geeksforgeeks.org/machine-learning/affinity-propagation-in-ml-to-find-the-number-of-clusters/) - identify data clusters by sending messages between data points, calculates optimal number of clusters automatically

#### Density-Based Methods
Density-Based (Model-Based) Methods define clusters as contiguous regions of high data density separated by areas of lower density
- [Mean-Shift Clustering](https://www.geeksforgeeks.org/machine-learning/ml-mean-shift-clustering/) - discovers clusters by moving points towards crowded areas
- [DBSCAN (Density-Based Spatial Clustering of Applications with Noise)](https://www.geeksforgeeks.org/machine-learning/dbscan-clustering-in-ml-density-based-clustering/) - Groups points with sufficient neighbors, labels sparse points as noise
- [OPTICS (Ordering Points To Identify the Clustering Structure)](https://www.geeksforgeeks.org/machine-learning/ml-optics-clustering-explanation/) - extends `DBSCAN` to handle varying densities

### Dimensionality Reduction
[Dimensionality Reduction](https://www.geeksforgeeks.org/machine-learning/dimensionality-reduction/) simplifies datasets by reducing features while keeping important information (often used to select features for other models)
- [PCA (Principal Component Analysis)](https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-pca/) - Reduces dimensions by transforming data into uncorrelated principal components.
- [ICA (Independent Component Analysis)](https://www.geeksforgeeks.org/machine-learning/ml-independent-component-analysis/)
- [t-SNE (t-distributed Stochastic Neighbor Embedding)](https://www.geeksforgeeks.org/machine-learning/ml-t-distributed-stochastic-neighbor-embedding-t-sne-algorithm/)
- [NMF (Non-negative Matrix Factorization)](https://www.geeksforgeeks.org/machine-learning/non-negative-matrix-factorization/) - Breaks data into non-negative parts to simplify representation.
- [Isomap](https://www.geeksforgeeks.org/machine-learning/isomap-a-non-linear-dimensionality-reduction-technique/) - Captures global data structure by preserving distances along a manifold.
- [LLE (Locally Linear Embedding)](https://www.geeksforgeeks.org/machine-learning/swiss-roll-reduction-with-lle-in-scikit-learn/) - Reduces dimensions while preserving the relationships between nearby points.
- [LDA (Linear Discriminant Analysis)](https://www.geeksforgeeks.org/machine-learning/ml-linear-discriminant-analysis/) - Reduces dimensions while maximizing class separability for classification tasks.

### Association Rule Mining
[Association Rule Mining](https://www.geeksforgeeks.org/machine-learning/association-rule/) discovers rules where the presence of one item in a dataset indicates the probability of the presence of another
- [Apriori Algorithm](https://www.geeksforgeeks.org/machine-learning/apriori-algorithm/) ([Implementation](https://www.geeksforgeeks.org/machine-learning/implementing-apriori-algorithm-in-python/)) - Finds patterns by exploring frequent item combinations step-by-step.
- [FP-Growth (Frequent Pattern-Growth)](https://www.geeksforgeeks.org/machine-learning/frequent-pattern-growth-algorithm/) - An Efficient Alternative to Apriori. It quickly identifies frequent patterns without generating candidate sets.
- [ECLAT (Equivalence Class Clustering and Bottom-Up Lattice Traversal)](https://www.geeksforgeeks.org/machine-learning/ml-eclat-algorithm/) - Uses intersections of itemsets to efficiently find frequent patterns.
- [Efficient Tree-based Algorithms](https://www.geeksforgeeks.org/dsa/introduction-to-tree-data-structure/) - Scales to handle large datasets by organizing data in tree structures.

## Reinforcement Learning
[Reinforcement Learning](https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/) learns from rewards by interacting with environment via trial and error

### Model-Based Methods
Model-Based Methods interact with a simulated environment
- [MDPs (Markov Decision Processes)](https://www.geeksforgeeks.org/machine-learning/markov-decision-process/)
- [Bellman Equation](https://www.geeksforgeeks.org/machine-learning/bellman-equation/)
- [Value Iteration Algorithm](https://www.geeksforgeeks.org/python/implement-value-iteration-in-python/)
- [Monte Carlo Tree Search](https://www.geeksforgeeks.org/machine-learning/ml-monte-carlo-tree-search-mcts/)

### Model-Free Methods
Model-Free Methods interact with the real environment
- [Q-Learning](https://www.geeksforgeeks.org/machine-learning/q-learning-in-python/)
- Deep Q-Learning
- [SARSA (State-Action-Reward-State-Action)](https://www.geeksforgeeks.org/machine-learning/sarsa-reinforcement-learning/)
- [Monte Carlo Methods](https://www.geeksforgeeks.org/python/monte-carlo-integration-in-python/)
- [Reinforce Algorithm](https://www.geeksforgeeks.org/machine-learning/reinforce-algorithm/)
- [Actor-Critic Algorithm](https://www.geeksforgeeks.org/machine-learning/actor-critic-algorithm-in-reinforcement-learning/)
- [A3C (Asynchronous Advantage Actor-Critic)](https://www.geeksforgeeks.org/machine-learning/asynchronous-advantage-actor-critic-a3c-algorithm/)

## Forecasting Models
Forecasting Models use past data to predict future trends (often time series problems)
- [ARIMA (Auto-Regressive Integrated Moving Average)](https://www.geeksforgeeks.org/r-language/model-selection-for-arima/)
- [SARIMA (Seasonal ARIMA)](https://www.geeksforgeeks.org/machine-learning/sarima-seasonal-autoregressive-integrated-moving-average/)
- [Exponential Smoothing (Holt-Winters)](https://www.geeksforgeeks.org/artificial-intelligence/exponential-smoothing-for-time-series-forecasting/)

## Semi-Supervised Learning
[Semi-Supervised Learning](https://www.geeksforgeeks.org/machine-learning/ml-semi-supervised-learning/) uses some labeled data with more unlabeled data
- [Self-Training](https://www.geeksforgeeks.org/machine-learning/self-training-in-semi-supervised-learning/) - The model is first trained on labeled data. It then predicts labels for unlabeled data, adding high-confidence predictions to the labeled set iteratively to refine the model.
- Co-Training - Two models are trained on different feature subsets of the data. Each model labels unlabeled data for the other, enabling them to learn from complementary views.
- Multi-View Training - A variation of co-training where models train on different data representations (e.g., images and text) to predict the same output.
- Graph-Based Models (Label Propagation) - Data is represented as a graph with nodes (data points) and edges (similarities). Labels are propagated from labeled nodes to unlabeled ones based on graph connectivity.
- GANs (Generative Adversarial Networks)
- [Few-Shot Learning](https://www.geeksforgeeks.org/machine-learning/few-shot-learning-in-machine-learning/) - a [meta-learning](https://www.geeksforgeeks.org/machine-learning/meta-learning-in-machine-learning/) process where you train the model to learn quickly from new and unseen data, so you don't have to train it with a bunch of data initially. So I guess it does some quick additional learning when you "inference" it later?

## Self-Supervised Learning
[Self-Supervised Learning](https://www.geeksforgeeks.org/machine-learning/self-supervised-learning-ssl/) generates its own labels from unlabeled data