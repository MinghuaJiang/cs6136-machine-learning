# cs6136_machine_learning
This is to record what I did in the machine learning course.

# hw2  linear regression, polynomial regression and ridge regression

# hw3  svm based salary classifier

Implemented a SVM Classifier using sklearn to predict the classification of the salary based on around 40k training set which includes about 15 features like countries, occupation, age, education, sex etc.

Preprocessed the data first including dealing with missing value like ?, encoding categorical values into numerical values, conducting one-hot encode for categorical values, scaling the features etc.

Implemented 3-fold cross validation for four different kernel function for SVM to find the best parameter. 

Achieved 85% test accuracy in a test set with around 10k records. 

# hw5  naive bayes movie review sentiment classifier

Implemented two different model of Naive Bayes Classifier for movie review sentiment analysis:
Multinomial and Multivariate Bernoulli

Used NLTK to preprocessed the data including stopword removing, stemming, vocabulary generation.

Created bag of word representation for data set.

Implemented the two model's training and testing without using sklearn and achieved the same test accuracy compared with using sklearn.

Achieved 80% test accuracy using Multinomial and 78% test accuracy using Multivariate Bernoulli
