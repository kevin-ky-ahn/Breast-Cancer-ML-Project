# Breast Cancer Diagnosis ML Project

## Project Motivation

I started this project as a way to re-familiarize myself with machine learning in Python using scikit-learn. I used kaggle.com's [breast cancer dataset](https://www.kaggle.com/datasets/wasiqaliyasir/breast-cancer-dataset), a popular dataset for practicing machine learning for real-life applications. The dataset contains ~600 rows of 32 columns (id, diagnosis, and the mean, standard error, and worst values for ten different features of each patient's tumor). 

I focused on refining models that could catch at least 98% of positive cases without too many false positives. I did not dive into the implications of the feature importances at different points as I do not have any relevant domain knowledge, and I did not have that in mind for the scope of this project.


## Methodology

I started with some initial exploration on the dataset and feature distributions, then looked at how a basic random forest performs. Before proceeding, I used the base dataset of means and standard errors to simulate sampling distributions for ~14000 samples of individual measurements of each feature. The intention was to create a dataset representing the initial measurements of each tumor, in order to train models for earlier diagnoses. This dataset only had one measurement of each tumor feature, referred to as the _mean value.

*Note: This sampling assumes each feature is normally distributed. It is also unclear how many different images are captured of each tumor to find the aggregates, and over what period of time. It may be that each aggregate is based on different captures of the tumor on the same day, in which case this simulated dataset would not be necessary*

I used 5 different machine learning algorithms - random forests, logistic regression, SVMs, K nearest neighbors classification, and neural networks. For each algorithm, I started by using GridSearch until I found the best hyperparameters, then I performed feature engineering until I settles on a model. I test this model by finding a favorable score threshold (based on minimizing false negatives in the training data), then test the results of the same threshold on the test dataset.

Then I took the logic used to generate that model, and recreated it on the original dataset (with the mean, standard error, and worst value of each feature). I train this model using the same hyperparamaters and feature transformations as the previous iteration of the model, then find the best threshold and observe the results on the test dataset.


## Expectations

I expected the final neural network to perform the best, even if it required much more tuning than the other models. I expected the random forest classifier to perform better than most other models with minimal tuning.


## Results

I made two exceptions to the general methodolgy: the random forest and the nerual network. For the random forest, I ran GridSearch a couple more times to make sure I was using the best hyperparameters. Each time I ran GridSearch, there was little difference in performance, so I generally chose the hyperparameters for the fastest model tuning. For most other models, when I saw little difference in hyperparameter performance, I did not run GridSearch any more times.

For the neural network, I knew the hyperparameters would be important to optimize, so I ran multiple iterations of trying different hyperparameters and hidden layer configurations (anywhere from 1 to 4 hidden layers, with numbers of nodes anywhere from 2/3 times the number of features to 10 times, as following conventional tips on hidden layer sizing were not helpful for the simulated data). Also, as the neural network based on the simulated dataset was not performing well, I performed another round of hyperparameter tuning and feature engineering on the original dataset, as well as including the standard error measurements in the feature transformations, which I did not do for other algorithms.

As stated, the goal of each model is to achieve a >=98% true positive rate while minimizing the false positive rate. The results of each model, on the test groups of the simulated dataset and the original dataset, are as follows:

| Model | Overall Accuracy | True Positive Rate | False Positive Rate |
| ------------- | ------------- | ------------- | ------------- |
| Random forest, simulated dataset | 93% | 99.2% | 14.8% |
| Logistic regression, simulated dataset | 95% | 99.8% | 17.9% |
| SVM, simulated dataset  | 95% | 99.5% | 12.1% |
| K nearest neighbors, simulated dataset | 93% | 99.0% | 15.2% |
| Neural network, simulated dataset  | 94% | 99.6% | 20.2% |

| Model | Overall Accuracy | True Positive Rate | False Positive Rate |
| ------------- | ------------- | ------------- | ------------- |
| Random forest, original dataset  | 96% | 98.0% | 10.8% |
| Logistic regression, original dataset  | 97% | 98.0% | 10.8% |
| SVM, original dataset  | 98% | 98.0% | 15.4% |
| K nearest neighbors, original dataset  | 97% | 100.0% | 13.8% |
| Neural network, original dataset  | 99% | 99.4% | 2.1% |

Most algorithms performed better on the original dataset, which isn't surprising as the addition of the worst measurement of each feature adds a lot of stratifying information. I was surprised just how much better the neural network performed on the original dataset, which seemed to indicate it prefers comprehensive information over large sample size for categorizing these tumors. While I put far more work into refining the original dataset model for the neural network than I did for the other algorithsm, I don't expect the other algorithms to reach quite that level of performance.

In future iterations of this or similar projects, I would probably create a function that allows me to run feature engineering and/or hyperparameter tuning in one cell, with prompts for inputs at each step. In this project, this would have allowed me to run multiple iterations of these processes in the same notebook without thousands of lines of repetitive code or having to create a massive, hard-to-navigate notebook, all while keeping record of all previous model iterations.

There was one particular case in the test group of the original dataset that was frequently a false negative: ID 855133. In the random forest classifier notebook, I compared the features of this case with the averages of other malignant cases, and it seemed to have a much smaller area and lower worst texture value than other malignant tumors, which would explain why it was consistently mislabeled. Again, I lack the domain knowledge to draw any informed conclusions from this information, but it seems to be a bit of an outlier without any significant impliciations on the dataset or breast cancer diagnosis.
