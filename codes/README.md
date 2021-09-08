# Codes Sequence

### **Do read the codes in the order below, so as to understand better on the codes and thought process.**


**1) Part 1 -  P. Statement, Background, Data Cleaning, NLP, and EDA**


This code is to state the problem statement and background of the project. 


Followed by processes that are being done to clean both the train and test dataframes, and then further preprocessing using the Natural Language Processing techniques is being done.

It is followed by multiple EDAs done to understand the dataset better.


**2) Part 2 - Modelling, Conclusions and Recommendations**


This code is to do modelling process on the data.

9 different models are being used.

1. Logistic Regression
2. Bernoulli Naive Bayes
3. Multinomial Naive Bayes
4. Random Forest
5. AdaBoost model
6. Gradient Boosting model
7. XGBoost model
8. Support Vector Machines
9. BERT Model (Ran using BERT Vectorizer)

Based on accuracy score, F1 score and AUC score, XGBoost Model using Count Vectorization has the highest of all the 3 scores, hence chosen for model evaluation.

Then model evaluation is then done and confusion matrix of the model is being printed. 

Lastly, conclusions and recommendations of this project.


**3) Appendix - Data Dictionary**

This code is the data dictionary for all the variables in the train_modified and test_modified dataset, for better understanding of what the items in the excel dataset mean.