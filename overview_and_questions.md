#DAND PS5 - Final Project - Questions & Answers

## Introduction
This is the final project for the Udacity Data Analyst Nanodegree problem set 5. It is based on a data set from the Intro to Machine Learning course which contains emails and financial data on the Enron fraud. The task is to create a model to identify persons of interest (POIs), these POIs are people that are for example convicted or suspects in the fraud case. Based on features in the data set the model has to correctly identify POIs, which in theory could also identify previously unknown POIs.

The final project consists of
- This document,  which will answer questions from the project.
- An iPython notebook, which explores the data set and various challenges in creating the final model
- poi_id.py, the final python script with the chosen model to create a classifier for the tester.py script.

## Questions and Answers

> Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The goal of this project is to find an algorithm that can classify persons of interest (POI) with great precision based on a set of features that vary from financial features like stock options and bonuses to the number of email sent to and received from known persons of interest. A person of interest meaning someone involved in the Enron fraud. The great thing about machine learning is that we can input all these features to train a model to predict who is or isn't a POI based on their features. These can be people that are different from the people in the data set that we trained the model on. In this case the data set was quite small with a little over 150 people, but we had quite a few features to work with. Although an algorithm can do lots of things that humans can't do, there are a few things like outlier detection that we do sometimes have to ourselves. In this case a visualisation and a further inspection of the PDF with names and financial data showed that there were two outliers that could safely be removed because they were not employees of Enron: "TOTAL" and "THE TRAVEL AGENCY IN THE PARK".


> What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

With regards to the features I used a SelectKBest algorithm in combination with a GridSearchCV to find the features and k number of features that had the most explanatory power. The chosen model had the following 15 features: (name, score, p-value)
- ('bonus', '30.729', '0.000')
- ('salary', '15.859', '0.000')
- ('to_POI_rate', '15.838', '0.000')
- ('total_value', '10.803', '0.001')
- ('shared_receipt_with_poi', '10.723', '0.001')
- ('total_stock_value', '10.634', '0.002')
- ('exercised_stock_options', '9.680', '0.002')
- ('total_payments', '8.959', '0.003')
- ('deferred_income', '8.792', '0.004')
- ('restricted_stock', '8.058', '0.006')
- ('long_term_incentive', '7.555', '0.007')
- ('loan_advances', '7.038', '0.009')
- ('from_poi_to_this_person', '4.959', '0.028')
- ('expenses', '4.181', '0.044')
- ('other', '3.204', '0.077')

Two of these features were features that I decided to add to the existing features. The first one is the to_POI_rate, a feature that defines the ratio of emails sent to a POI out of the total number of emails sent. I decided to add this feature to see if email contact was an indicator, but I decided to correct for the number of emails sent by looking at the total. Using a different scale, the ratio instead of the absolute number of emails, was helpful as a new feature.  
The second feature is total_value. This feature is sum of all financial features, payments as well as stocks. I found that most people would have missing values for a lot of the financial features. Not a strange thing considering the various ways in which someone can be compensated. Most people however had at least one method of compensation. I wondered what the effect would be of using one single number, the sum of all compensation, as a feature.

> What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

I ended up using a Decision Tree classifier. I thought it interesting that the Naive Bayes classifier did surprisingly well. I think the explanation for that is that it works quite well because it takes into account conditional probability and can therefor deal with the imbalanced nature of our data set quite well. I was expecting the Support Vector Machine classifier (SVC) to perform better than it ended up doing (an F1-score of .12) because of its ability to deal with complexity better. The biggest problem I ended up having was the realisation that I hadn't scaled my features properly for the SVC to deal with them. I ended up trying a MinMaxScaler and a StandardScaler —which does a z-score normalisation— for use with the SVC and got a bit better results through that. This also gave me the idea to add the scaler to my pipeline and use it for the GridSearchCV to try on all the different classifiers. This ended up getting even better results for the Decision Tree and got its precision, recall and F1 scores all above .30.

> What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

The parameters for a Decision Tree classifier tune the model's complexity. A decision tree where the size of the leaves is not regulated can grow into an overcomplex model thereby overfitting on the training data. I used stratified shuffle to control for the small data set, but this can result in wildly different trees. To resolve this problem I cranked up the number of iterations on the stratified shuffle, choosing a minimal number of elements to make a split on —splitting on just two elements will quickly lead to an overly complex tree—, and similarly a minimum number of elements to be on a leaf of the tree.

> What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

Validation is testing the performance of the model you use. Obviously if you would use the same data for training and testing —validating— your model you would end up overfitting the model. One other problem I encountered in this data set is that it is quite small, only about 150 people. That means that a randomised test will often be way off, because maybe too many or too little POI's are selected in the test. The other problem is that the number of POI's is also very small. That means that accuracy will not be a good score for validating our model. Just by classifying everyone as non-POI we would already have an accuracy score of .875 (126 non-POI's out of 144 people). That means we need a different scoring mechanism. For selecting the final model I choose the F1 score, which combines precision and recall.

> Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

The F1-score I used for selecting the final model is based on precision and recall. Precisions meaning the number of true positive identifications divided by true and false positive identifications combined. Recall, on the other hand, means the number of true positives divided by true positives and false negatives combined. The nice thing about the F1-score is that it combines precision (p) and recall (r) in a simple formula: 2*(pr/(p+r)). When putting the final model through tester.py I ended up with the following scores:
- Precision: 0.36183
- Recall: 0.30900
- F1: 0.33333


## References
#### Articles
- Sebastian Raschka, *About Feature Scaling and Normalization: and the effect of standardization for machine learning algorithms*, http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

- Adam B. Shinn, *Grid Searching in all the Right Places*, http://abshinn.github.io/python/sklearn/2014/06/08/grid-searching-in-all-the-right-places/

- Katie Malone, *Workflows in Python: Using Pipeline and GridSearchCV for More Compact and Comprehensive Code*, https://civisanalytics.com/blog/data-science/2016/01/06/workflows-python-using-pipeline-gridsearchcv-for-compact-code/

- Ram Narasimhan, *A simple explanation of Naive Bayes Classification*, http://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification (This is a wonderful and thorough answer on conditional probability and Naive Bayes classifiers)

#### Forum discussions
- https://discussions.udacity.com/t/getting-started-with-final-project/170846/6?u=dumkydewilde)
- https://discussions.udacity.com/t/confusion-about-order-of-operations-in-poi-id-file-feature-selection/182193/2
- https://discussions.udacity.com/t/new-feature-addition/7109
- https://discussions.udacity.com/t/how-to-find-out-the-features-selected-by-selectkbest/45118

#### Other
The scikit-learn documentation was very helpful for implementing the pipeline, the feature selection, the grid search and understanding the parameters of the models I explored. For example:
- http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html
- http://scikit-learn.org/stable/modules/pipeline.html
