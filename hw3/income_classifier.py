# Starting code for CS6316/4501 HW3, Fall 2016
# By Weilin Xu

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.metrics import accuracy_score
import random
import logging
import os.path
from sklearn.externals import joblib
import pickle
import threading


# Attention: You're not allowed to use the model_selection module in sklearn.
#            You're expected to implement it with your own code.
# from sklearn.model_selection import GridSearchCV

class SvmIncomeClassifier:
    def __init__(self):
        random.seed(0)

    def load_data(self, csv_fpath):

        logging.info("Loading data:" + csv_fpath)

        col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                       'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                       'hours-per-week', 'native-country']
        col_names_y = ['label']

        numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                          'hours-per-week']
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                            'race', 'sex', 'native-country']

        # 1. Data pre-processing.
        # Hint: Feel free to use some existing libraries for easier data pre-processing.
        names = tuple(col_names_x + col_names_y)
        data = np.loadtxt(csv_fpath, dtype={'names': names, 'formats': tuple(
            ['f8' if i in numerical_cols else 'S24' for i in names])},
                          delimiter=', ')

        y_val = data[col_names_y[0]]
        enc_label = LabelEncoder()
        enc_label.fit(y_val)

        y_val = enc_label.transform(y_val)
        x_val = np.zeros((data.size, len(col_names_x)))

        for i in range(0, len(col_names_x)):
            if col_names_x[i] in numerical_cols:
                x_val[:, i] = data[col_names_x[i]]
            else:
                if os.path.isfile(col_names_x[i] + ".pkl"):
                    enc_label = joblib.load(col_names_x[i] + ".pkl")
                else:
                    enc_label = LabelEncoder()
                    enc_label.fit(data[col_names_x[i]])
                    joblib.dump(enc_label, col_names_x[i]+".pkl")

                counter = Counter(list(data[col_names_x[i]]))
                mode = counter.most_common(1)[0][0]
                for j in range(0, data[col_names_x[i]].size):
                    if data[col_names_x[i]][j] == '?':
                        data[col_names_x[i]][j] = mode

                le_classes = enc_label.classes_.tolist()
                for index in range(0, len(data[col_names_x[i]])):
                    if data[col_names_x[i]][index] not in enc_label.classes_:
                        le_classes.append(data[col_names_x[i]][index])
                enc_label.classes_ = le_classes

                data[col_names_x[i]] = enc_label.transform(data[col_names_x[i]])
                x_val[:, i] = data[col_names_x[i]].astype('f8')

        if os.path.isfile("hot_encoder.pkl"):
            hot_encoder = joblib.load("hot_encoder.pkl")
        else:
            hot_encoder = OneHotEncoder(categorical_features=[col_names_x.index(i) for i in categorical_cols],
                                        handle_unknown='ignore')
            hot_encoder.fit(x_val)
            joblib.dump(hot_encoder, "hot_encoder.pkl")

        x_val = hot_encoder.transform(x_val).toarray()

        min_max_scaler = MinMaxScaler()
        x_val = min_max_scaler.fit_transform(x_val)
        return x_val, y_val

    def cv_worker(self, param, x_train, y_train, clf_map):
        k_folder = 3
        splitted_x = np.array_split(x_train, k_folder)
        splitted_y = np.array_split(y_train, k_folder)
        data_rows = len(y_train)
        data_columns_x = len(x_train[0])
        data_columns_y = 1

        train_score = 0
        validation_score = 0
        svc = SVC(**param)
        for i in range(0, k_folder):
            validation_x = splitted_x[i]
            validation_y = splitted_y[i]
            validation_rows = len(splitted_y[i])
            train_rows = data_rows - validation_rows

            train_x = np.zeros((train_rows, data_columns_x))
            train_y = np.zeros((train_rows, data_columns_y))

            current = 0
            for j in range(0, i):
                rows = len(splitted_y[j])
                for k in range(0, rows):
                    train_x[current + k] = splitted_x[j][k]
                    train_y[current + k] = splitted_y[j][k]
                current += rows

            for j in range(i + 1, k_folder):
                rows = len(splitted_y[j])
                for k in range(0, rows):
                    train_x[current + k] = splitted_x[j][k]
                    train_y[current + k] = splitted_y[j][k]
                current += rows
            train_y = train_y.flatten()
            logging.info("Learning Model under folder %s and params %s" % (i + 1, str(param.items())))
            svc.fit(train_x, train_y)
            logging.info("Running prediction under folder %s and params %s" % (i + 1, str(param.items())))
            train_predict_y = svc.predict(train_x)
            validation_predict_y = svc.predict(validation_x)
            train_score += accuracy_score(train_y, train_predict_y)
            validation_score += accuracy_score(validation_y, validation_predict_y)
        validation_score /= k_folder
        train_score /= k_folder
        clf_map[validation_score] = svc
        logging.info("train accuracy %.4f with params %s" % (train_score, str(param.items())))
        logging.info("validation accuracy %.4f with params %s" % (validation_score, str(param.items())))

    def train_and_select_model(self, csv):
        if os.path.isfile(csv+"_training_model.pkl"):
            logging.info("Found existing model trained with file: %s" % csv)
            with open(csv+"_training_model.pkl", 'rb') as model_file:
                clf_map = pickle.load(model_file)
            best_score = max(clf_map.keys())
            best_clf = clf_map[best_score]
            return best_clf, best_score
        logging.info("Training model:" + csv)
        x_train, y_train = self.load_data(csv)
        # 2. Select the best model with cross validation.
        # Attention: Write your own hyper-parameter candidates.
        param_set = [
            {'kernel': 'rbf', 'C': 20, 'gamma': 0.03},
            {'kernel': 'linear', 'C': 1},
            {'kernel': 'poly', 'C': 1, 'degree': 1, 'gamma': 1, 'coef0': 1},
            {'kernel': 'sigmoid', 'C': 0.0001, 'gamma': 1, 'coef0': 1},
        ]
        clf_map = dict()
        for param in param_set:
            t = threading.Thread(target=self.cv_worker, args=(param, x_train, y_train, clf_map,))
            t.start()

        logging.debug('Waiting for worker threads')
        main_thread = threading.currentThread()
        for t in threading.enumerate():
            if t is not main_thread:
                t.join()

        best_score = max(clf_map.keys())
        best_clf = clf_map[best_score]
        with open(csv+'_training_model.pkl', 'wb') as output:
            pickle.dump(clf_map, output)
        return best_clf, best_score

    def predict(self, test_csv, model):
        x_test, _ = self.load_data(test_csv)
        predict = model.predict(x_test)
        return predict

    def output_results(self, predict):
        # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
        # Hint: Don't archive the files or change the file names for the automated grading.
        with open('predictions.txt', 'w') as f:
            for pred in predict:
                if pred == 0:
                    f.write('<=50K\n')
                else:
                    f.write('>50K\n')


if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    logging.basicConfig(filename='hw3.log', level=logging.INFO, format='%(asctime)s %(message)s')
    clf = SvmIncomeClassifier()
    trained_model, cv_score = clf.train_and_select_model(training_csv)
    logging.info("The best model was scored %.4f" % cv_score)
    predictions = clf.predict(testing_csv, trained_model)
    clf.output_results(predictions)
