# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone Project: Fake News Classification

## Problem Statement

This projects aim to use data science methods (classification modellings) to predict whether a news article is fake or real, with prediction having the higher the accuracy, F1 and AUC Score the better (as close to 1 as possible), so as to enable US citizens to have a better capability to differentiate between fake or real news.

## Background



**What is fake news?**

Fake news are false or misleading information presented as news. The purpose of fake news is to damage the reputation of a person or entity, or making money through advertising revenue.

**Danger of fake news problem**

According to [NewsGuard](https://www.axios.com/unreliable-news-sources-social-media-engagement-297bf046-c1b0-4e69-9875-05443b1dca73.html?utm_campaign=organic&utm_medium=socialshare&utm_source=twitter), in the year 2020, 2.8b out of 16.3b (17%) of social media engagement among the top 100 news sources on social media came from unrealiable source (i.e fake news).

A report published by [CHEC](https://s3.amazonaws.com/media.mediapost.com/uploads/EconomicCostOfFakeNews.pdf) in the year 2019 estimated the global economic cost of fake news to be $78 billion per year which includes direct and indirect cost incurred in the following areas: stock markets, media, reputation management, election campaigns, financial information and healthcare.

**Fake News in US**

In US, most of the fake news revolve around politics. 

In the 2016 presidential election, fake news as a term have become a buzzword  (Egelhofer and Lechele, 2019; Egelhofer, et al., 2020; Farkas and Schou, 2018). A [buzzfeed](https://abcnews.go.com/Technology/fake-news-stories-make-real-news-headlines/story?id=43845383) analysis found that the top fake news stories in that election received more engagement on social media than top stories from major media outlets. 

After Donald Trump won the 2016 election, there are [studies](https://www.washingtonpost.com/news/the-fix/wp/2018/04/03/a-new-study-suggests-fake-news-might-have-won-donald-trump-the-2016-election/) being conducted on his win, and fake news is attributed to be one of the reasons of his win. Some of the fake news being spread around that time are: Hillary Clinton approved weapons sales to ISIS, Hillary Clinton was in “very poor health due to serious illness”.

In the 2020 presidential election, multiple [fake news](https://edition.cnn.com/business/live-news/election-2020-misinformation/index.html) being generated to portray that the election was fraud and was stolen from Republicans and Donald Trump. Some examples of these are: Burning of ballot boxes with votes belonging to Trump, Claims that dead people went voting, etc.

As a result of believing in fake news, and believing that Trump actually won the election and stolen, a mob of supporters of Trump violently attacked the United States Capitol in Washington, D.C. on January 6, 2021.

## Data Dictionary
Data Dictionary for all datasets are being stored in this link below:
[Data_Dictionary](http://localhost:8888/notebooks/OneDrive/Desktop/DSI23%20-%20Capstone/Capstone/codes/Appendix%20-%20Data%20Dictionary.ipynb)


## Data Used in Analysis
1 main dataset used which records the title and text of the news articles in US. There are around 25,000 entries of articles, with time range of around the 2016 Presidential Election.

## Workflow Outline
1. Obtaining the Dataset
2. Data Cleaning/Preprocessing
3. EDA
4. Modelling and Evaluations
5. Recommendations

## Data Cleaning/Preprocessing
Following steps are being done to clean and preprocess the data.
1. Data Cleaning
- Imputation of null values with “-” to keep the other columns info
- Remove duplicate entries
2. Natural Language Processing Techniques
- Removing punctuations
- Tokenizing
- Remove stopwords
- Done lemmatization
3. Vectorization
- Count Vectorization
- TF-IDF Vectorization

## Exploratory Data Analysis (EDA)
The following insights were found during EDA:  

**1. Length of News Articles by Word Count**
- There are a few more fake news articles having > 10,000 words.
- However, based on distribution, a lot more fake news articles have 0-500 words compared to real news.
- Most of real/fake  news  articles  have  length  of  <  2,000 words.

**2. Frequently Appeared 1-Word**

By using N-Gram, top-20 1-word that appears frequently in both real and fake news articles are obtained.

Not much insight could be gathered here except that the words appeared in both shown that these articles are very related to 2016 US Presidential Election with words such as "trump", "president", "hillary", "clinton", "american", "election".

**3. Frequently Appeared 2-Words**

By using N-Gram, top-20 12-words that appears frequently in both real and fake news articles are obtained.
- A lot of the 2-words combination appear in both the real and fake news articles. Hence, unable to use the word frequency to properly classify fake or real news.
- However,  can  observe  that  a  lot  of  fake articles are targeting on Hillary Clinton or Middle East foreign policy.

**4. Sentiment Analysis of News Articles**

Using Sentiment Analysis Compound Score, sentiment  of  both  real  and  fake  articles are obtained.

Compound Score < (-0.05): Negative Sentiment, -0.05 to 0.05: Neutral, > 0.05: Positive Sentiment

- Within the real news articles, the proportion of positive to negative is around 60:40, which is more positive than the fake news articles with proportion of around 50:50.
- However, the sentiment is not too binary (real == +ve, fake == -ve), hence sentiment analysis is not a good way to classify fake/real news.

## Model Evaluations
The evaluation metrics used are as follows:

**Test Accuracy**:
- Indicator of how accurate is the prediction.
- Good model = High test accuracy, closer to 100%

**F1-score**:
- Weighted average of Precision-Recall score, used to compare the performance of classifiers
- Good model = High F1-score, closer to 100%

**Area Under Curve (AUC)**:
- Ability to distinguish between Positive (fake news articles) and Negative (real news articles)
- Good model = High AUC, closer to 1

**Train/Test Accuracy**:
- Indicator of overfitting or underfitting of the model.
- Train - Test > 0 = Overfitting
- Train - Test < 0 = Underfitting

**Models**
The following models were ran and evaluated:
1. Logistic Regression
2. Bernoulli Naive Bayes
3. Multinomial Naive Bayes
4. Random Forest
5. AdaBoost model
6. Gradient Boosting model
7. XGBoost model
8. Support Vector Machines
9. BERT Model (Ran using BERT Vectorizer)

 

|Different Model|Types of Vectorization|Test Accuracy|Train Accuracy|Train-Test Accuracy|Precision|Recall|F1-score|AUC|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**Baseline**|-|0.53|-|-|-|-|-|-|
|**Logistic Regression**|**Count Vectorization**|0.95|0.96|0.01|0.96|0.96|0.96|0.990|
||**TFIDF Vectorization**|0.94|0.95|0.01|0.96|0.96|0.96|0.990|
|**Bernoulli Naive Bayes**|**Count Vectorization**|0.76|0.76|0|0.79|0.77|0.77|0.850|
||**TFIDF Vectorization**|0.76|0.76|0|0.78|0.77|0.77|0.850|
|**Multinomial Naive Bayes**|**Count Vectorization**|0.91|0.91|0|0.91|0.91|0.91|0.965|
||**TFIDF Vectorization**|0.90|0.91|0.01|0.92|0.91|0.91|0.975|
|**Random Forest**|**Count Vectorization**|0.93|0.94|0.01|0.94|0.94|0.94|0.989|
||**TFIDF Vectorization**|0.91|0.93|0.02|0.93|0.93|0.93|0.984|
|**AdaBoost Model**|**Count Vectorization**|0.95|0.96|0.01|0.96|0.96|0.96|0.992|
||**TFIDF Vectorization**|0.95|0.96|0.01|0.95|0.95|0.95|0.991|
|**Gradient Boost Model**|**Count Vectorization**|0.96|0.96|0|0.96|0.96|0.96|0.994|
||**TFIDF Vectorization**|0.95|0.96|0.01|0.96|0.96|0.96|0.993|
|**XGBoost Model**|**Count Vectorization**|0.97|0.98|0.01|0.98|0.98|0.98|0.998|
||**TFIDF Vectorization**|0.96|0.97|0.01|0.97|0.97|0.97|0.997|
|**Support Vector Machines**|**Count Vectorization**|0.82|0.74|-0.08|0.76|0.70|0.69|0.868|
||**TFIDF Vectorization**|0.85|0.85|0|0.86|0.86|0.86|0.933|
|**BERT Model**|**BERT Vectorization**|0.95|0.95|0|0.95|0.95|0.95|0.948|

Above is a table showing the summary of the scores of all the different models. 

The best model selected is the **XGBoost model using Count Vectorization**.  

This is because XGBoost model using Count Vectorization have the highest test accuracy score out of all models, as well as having the highest F1 score out of all models, and the highest AUC score out of all models. With regards to the difference between train accuracy and test accuracy, XGBoost model using Count Vectorization have train - test accuracy > 0, hence it is overfitting. However, the difference is minimal (~0.01), hence the overfitting is not that severe.


## Conclusion
XG Boost Model using Count Vectorizer have the best performance.
- Accurate classification ~ 97% of the time; 5,723 out of 5,853 predictions
- AUC score ~ 1%

## Recommendation

**1. Deploy the model for use as a self fact-check system in testing with actual articles**

Given the strong performance of model (97% accuracy score, 98% F1 score, ~1 AUC Score), US citizens can benefit from deploying the model for purpose of self fact-check of political news. 

**2. Retrain the model periodically with new political news articles.**

As time passes, there would be a need to retrain the model with new and more relevant political news articles. This is to continuing to ensure that the model can continually maintained a high performance.

**3. Explore the expansion of the model to cover other news area.**

Currently, the model only includes news articles from politics. In the near future, it would be better if the model can expand to include fact-check of other areas: healthcare, education, sports, etc.
