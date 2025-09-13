# Machine Learning
[Machine Learning](https://www.geeksforgeeks.org/machine-learning/machine-learning/) is a branch of Artificial Intelligence that focuses on *models* and [algorithms](https://www.geeksforgeeks.org/machine-learning/machine-learning-algorithms/) that let computers learn from data and improve from previous experience without being explicitly programmed. There are many [types](https://www.geeksforgeeks.org/machine-learning/types-of-machine-learning/) of machine learning.


## Types of Machine Learning
- [Supervised Learning](https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/) - Use labeled data
    - [Classification](https://www.geeksforgeeks.org/machine-learning/getting-started-with-classification/) - Predict *categorical* (discrete) values
        - Linear Classifiers
            - [Logistic Regression](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/) - Draws a sigmoid curve, predicts 0 or 1 if above or below curve.
        - Non-Linear Classifiers
            - KNN (K-Nearest Neighbors)
            - Naive Bayes
        - Linear or Non-Linear Classifiers
            - SVM (Support Vector Machine)
    - [Regression](https://www.geeksforgeeks.org/machine-learning/regression-in-machine-learning/) - Predict continuous numerical values
        - [Linear Regression](https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/) - fit a straight line to the data
        - Polynomial Regression
        - Ridge Regression
        - Lasso Regression
    - Both - can be used for either Regression or Non-Linear Classification
        - Building Blocks - these models are often used as building blocks for Ensemble Methods
            - Decision Tree
        - Ensemble Learning - Combine multiple simple models into one better model
            - Bagging (Bootstrap Aggregating) Method - Train models independently on different subsets of the data, then combine their predictions
                - Random Forest
                - Random Subspace Method
            - Boosting Method - Train models sequentially, each model focusing on errors of prior models, then do weighted combination of their predictions
                - Adaptive Boosting (AdaBoost)
                - Gradient Boosting
                - Extreme Gradient Boosting (XGBoost)
                - CatBoost
            - [Stacking (Stacked Generalization) Method](https://machinelearningmastery.com/implementing-stacking-scratch-python/) - train multiple different models (often different types), use predictions as inputs to final "meta-model"
- [Unsupervised Learning](https://www.geeksforgeeks.org/machine-learning/unsupervised-learning/) - Use unlabeled data
    - Clustering - Group data into clusters based on similarity
        - Centroid-Based Methods
            - K-Means Clustering
            - Elbow Method for optimal value of k in KMeans
            - K-Means++ Clustering
            - K-Mode Clustering
            - Fuzzy C-Means (FCM) Clustering
        - Distribution-Based Methods
            - Gaussian Mixture Models
            - Expectation-Maximization Algorithm
            - Dirichlet Process Mixture Models (DPMMs)
        - Connectivity-Based Methods
            - Hierarchical Clustering
            - Agglomerative Clustering
            - Divisive Clustering
            - Affinity propagation
        - Density-Based Methods
            - Mean-Shift
            - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
            - OPTICS (Ordering Points To Identify the Clustering Structure)
    - Dimensionality Reduction - Simplify datasets by reducing features while keeping important information (often used to select features for other models)
        - PCA (Principal Component Analysis)
        - Independent Component Analysis (ICA)
        - t-distributed Stochastic Neighbor Embedding (t-SNE)
        - Non-negative Matrix Factorization (NMF)
        - Isomap
        - Locally Linear Embedding (LLE)
    - Association Rule - Discover rules where the presence of one item in a dataset indicates the probability of the presence of another
        - Apriori
        - Equivalence Class Clustering and Bottom-Up Lattice Traversal (ECLAT)
        - Frequent Pattern Growth (FP-Growth)
- [Reinforcement Learning](https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/) - Learn from rewards by interacting with environment via trial and error
    - Model-Based Methods - Interact with a simulated environment
        - Markov Decision Processes (MDPs)
        - Bellman Equation
        - Value Iteration Algorithm
        - Monte Carlo Tree Search
    - Model-Free Methods - Interact with the real environment
        - Q-Learning
        - Deep Q-Learning
        - State-Action-Reward-State-Action (SARSA)
        - Monte Carlo Methods
        - Reinforce Algorithm
        - Actor-Critic Algorithm
        - Asynchronous Advantage Actor-Critic (A3C)
- Forecasting Models - Use past data to predict future trends (often time series problems)
    - ARIMA (Auto-Regressive Integrated Moving Average)
    - SARIMA (Seasonal ARIMA)
    - Exponential Smoothing (Holt-Winters)
- [Semi-Supervised Learning](https://www.geeksforgeeks.org/machine-learning/ml-semi-supervised-learning/) - Use some labeled data with more unlabeled data
    - Graph-Based Semi-Supervised Learning
    - Label Propagation
    - Co-Training
    - Self-Training
    - Generative Adversarial Networks (GANs)
- [Self-Supervised Learning](https://www.geeksforgeeks.org/machine-learning/self-supervised-learning-ssl/) - Generates its own labels from unlabeled data

## Ensemble Learning
- I'm not sure the best place to put this, so I'm putting it here. [Ensemble Learning](https://www.geeksforgeeks.org/machine-learning/a-comprehensive-guide-to-ensemble-learning/) is about combining many

## Supervised Learning
Uses labeled data
- [GeeksForGeeks Supervised Machine Learning](https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/)

### Classification
- [GeeksForGeeks Getting Started With Classification](https://www.geeksforgeeks.org/machine-learning/getting-started-with-classification/)
- [Machine Learning Mastery Classification Algorithms](https://machinelearningmastery.com/5-essential-classification-algorithms-explained-beginners/).

#### Logistic Regression
- [GeeksForGeeks Understanding Logistic Regression](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/)
Though it has "regression" in  the name, it's for classification

#### Support Vector Machine (SVM)
- [GeeksForGeeks Support Vector Machine Algorithm](https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/)

#### Random Forest
- [GeeksForGeeks Random Forest Regression](https://www.geeksforgeeks.org/machine-learning/random-forest-regression-in-python/)

#### Decision Tree
- [GeeksForGeeks Decision Tree](https://www.geeksforgeeks.org/machine-learning/decision-tree/)

#### K-Nearest Neighbors (KNN)
- [GeeksForGeeks K-Nearest Neighbors](https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/)
- [My DataCamp Classification Notebook](../datacamp/1_supervised_learning_scikit_learn/1_classification.ipynb)
- binary output like "will this customer leave (`1`) or stay (`0`) based on account `age` and `customer service call count`"
- sets the result for the target based on nearby points (must be an odd number of points to prevent a "tie")
- it determines "nearness" by mapping the inputs in vector space and computing the length of the vector between each training data point and test data point

#### Naive Bayes
- [GeeksForGeeks Naive Bayes Classifiers](https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/)

### Regression
- [GeeksForGeeks Types of Regression Techniques](https://www.geeksforgeeks.org/machine-learning/types-of-regression-techniques/)
- [GeeksForGeeks Regression in Machine Learning](https://www.geeksforgeeks.org/machine-learning/regression-in-machine-learning/)
Predict a continuous output variable

#### Linear Regression
- [GeeksForGeeks Linear Regression](https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/)

#### Polynomial Regression
- [GeeksForGeeks Polynomial Regression Algorithm](https://www.geeksforgeeks.org/videos/polynomial-regression-algorithm-machine-learning/)

#### Ridge Regression
- [GeeksForGeeks Ridge Regression Algorithm](https://www.geeksforgeeks.org/videos/lasso-ridge-regression-algorithm-machine-learning/)

#### Lasso Regression
- [GeeksForGeeks Lasso Regression Algorithm](https://www.geeksforgeeks.org/videos/lasso-ridge-regression-algorithm-machine-learning/)

#### Decision Tree
- [GeeksForGeeks Decision Tree Introduction Example](https://www.geeksforgeeks.org/machine-learning/decision-tree-introduction-example/)

#### Random Forest
- [GeeksForGeeks Random Forest Regression](https://www.geeksforgeeks.org/machine-learning/random-forest-regression-in-python/)


## Unsupervised Learning
- [GeeksForGeeks Unsupervised Learning](https://www.geeksforgeeks.org/machine-learning/unsupervised-learning/)

### Clustering
- [GeeksForGeeks Clustering](https://www.geeksforgeeks.org/machine-learning/clustering-in-machine-learning/)

#### K-Means Clustering
- [GeeksForGeeks K-Means Clustering Introduction](https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/)

#### Mean-Shift
- [GeeksForGeeks Mean-Shift Clustering](https://www.geeksforgeeks.org/machine-learning/ml-mean-shift-clustering/)

#### Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
- [GeeksForGeeks DBSCAN Clustering](https://www.geeksforgeeks.org/machine-learning/dbscan-clustering-in-ml-density-based-clustering/)

#### Principal Component Analysis (PCA)
- [GeeksForGeeks Principal Component Analysis](https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-pca/)

#### Independent Component Analysis (ICA)
- [GeeksForGeeks Independent Component Analysis](https://www.geeksforgeeks.org/machine-learning/ml-independent-component-analysis/)

### Association Rule
- [GeeksForGeeks Association Rule](https://www.geeksforgeeks.org/machine-learning/association-rule/)

#### Apriori
- [GeeksForGeeks Apriori Algorithm](https://www.geeksforgeeks.org/machine-learning/apriori-algorithm/)

#### Equivalence Class Clustering and Bottom-Up Lattice Traversal (ECLAT)
- [GeeksForGeeks ECLAT Algorithm](https://www.geeksforgeeks.org/machine-learning/ml-eclat-algorithm/)

#### Frequent Pattern Growth (FP-Growth)
- [GeeksForGeeks FP-Growth Algorithm](https://www.geeksforgeeks.org/machine-learning/frequent-pattern-growth-algorithm/)


## Reinforcement Learning
- [GeeksForGeeks What Is Reinforcement Learning](https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/)

### Q-Learning
[GeeksForGeeks Q-Learning](https://www.geeksforgeeks.org/machine-learning/q-learning-in-python/)

### Deep Q-Learning
[GeeksForGeeks Deep Q-Learning](https://www.geeksforgeeks.org/deep-learning/deep-q-learning/)

### State-Action-Reward-State-Action (SARSA)
[GeeksForGeeks State-Action-Reward-State-Action (SARSA)](https://www.geeksforgeeks.org/machine-learning/sarsa-reinforcement-learning/)


## Semi-Supervised Learning
- [GeeksForGeeks Semi-Supervised Learning](https://www.geeksforgeeks.org/machine-learning/ml-semi-supervised-learning/)

### Graph-Based Semi-Supervised Learning
use a graph to represent relationships between data points, then propagate labels from labeled to unlabeled data points.

### Label Propagation
iteratively propagate labels from labeled to unlabeled data points

### Co-Training
no link

### Self-Training
no link

### Generative Adversarial Networks (GANs)
- [GeeksForGeeks Generative Adversarial Networks (GANs)](https://www.geeksforgeeks.org/deep-learning/generative-adversarial-network-gan/)

## Self-Supervised Learning
