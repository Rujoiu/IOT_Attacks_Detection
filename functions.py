import numpy as np
import pandas as pd
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pickle


# The following 3 methods are used to write in the top level window
def add_new(window, text):
    window.add_new_text(text)


def add_new_ti(window, text):
    window.add_new_title(text)


def add_new_err(window, text):
    window.add_new_error(text)


# Loads data from the csv files and creates labels, given a device
# Deals with +/- infity values and NaN values
def functia_desteapta(device):
    folder = "devices"
    m_folder = "mirai_attacks"
    g_folder = "gafgyt_attacks"
    benign_file = "benign_traffic.csv"
    extension = ".csv"

    benign_path = os.path.join(folder, device, benign_file)
    mirai_path = os.path.join(folder, device, m_folder)
    gafgyt_path = os.path.join(folder, device, g_folder)

    data_benign = pd.read_csv(benign_path)
    data_benign['label'] = 'Benign'
    device_data = data_benign

    if (device != "Ennio_Doorbell" and device != "Samsung_SNH"):
        for file in os.listdir(mirai_path):
            file_name = os.path.join(mirai_path, file)
            data = pd.read_csv(file_name)
            label = file.partition('.')[0]
            data['label'] = label
            device_data = pd.concat([data, device_data], ignore_index=True)

    for file in os.listdir(gafgyt_path):
        file_name = os.path.join(gafgyt_path, file)
        data = pd.read_csv(file_name)
        label = file.partition('.')[0]
        data['label'] = label
        device_data = pd.concat([data, device_data], ignore_index=True)

    device_data = device_data.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    device_data.dropna()

    return device_data


# Loads data from csv files for all the devices and creates labels
# Deals with +/- infity values and NaN values
def functia_desteapta_all(window):
    devices_list = ["Damnini_Doorbell", "Ecobee_Thermostat", "Ennino_Doorbell", "Philips_B120N10_Baby_Monitor",
                    "Provision_PT_737E_Security_Camera", "Provision_PT_838_Security_Camera", "Samsung_SNH",
                    "SimpleHome_1002_SecurityCamera", "SimpleHome_1003_SecurityCamera"]
    folder = "devices"
    m_folder = "mirai_attacks"
    g_folder = "gafgyt_attacks"
    benign_file = "benign_traffic.csv"
    extension = ".csv"
    for device in devices_list:
        try:
            print(device)
            benign_path = os.path.join(folder, device, benign_file)
            mirai_path = os.path.join(folder, device, m_folder)
            gafgyt_path = os.path.join(folder, device, g_folder)

            data_benign = pd.read_csv(benign_path)
            data_benign['label'] = 'Benign'
            device_data = data_benign

            if (device != "Ennino_Doorbell" and device != "Samsung_SNH"):
                for file in os.listdir(mirai_path):
                    file_name = os.path.join(mirai_path, file)
                    data = pd.read_csv(file_name)
                    label = file.partition('.')[0]
                    data['label'] = label
                    device_data = pd.concat([data, device_data], ignore_index=True)

            for file in os.listdir(gafgyt_path):
                file_name = os.path.join(gafgyt_path, file)
                data = pd.read_csv(file_name)
                label = file.partition('.')[0]
                data['label'] = label
                device_data = pd.concat([data, device_data], ignore_index=True)

            device_data = device_data.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
            device_data.dropna()
            if (device == "Damnini_Doorbell"):
                all_data = device_data
            else:
                all_data = pd.concat([all_data, device_data], ignore_index=True)

        except FileNotFoundError:
            print("Unable to find the file for: " + device)
            print("Check if the dataset is complete and try again")
            add_new_err(window, "Unable to find the file for: " + device)
            add_new_err(window, "Check if the dataset is complete and try again")

    return all_data


def new_functia_desteapta_all():
    devices_list = ["Damnini_Doorbell", "Ecobee_Thermostat", "Ennino_Doorbell", "Philips_B120N10_Baby_Monitor",
                    "Provision_PT_737E_Security_Camera", "Provision_PT_838_Security_Camera", "Samsung_SNH",
                    "SimpleHome_1002_SecurityCamera", "SimpleHome_1003_SecurityCamera"]
    folder = "devices"
    m_folder = "mirai_attacks"
    g_folder = "gafgyt_attacks"
    benign_file = "benign_traffic.csv"
    extension = ".csv"
    for device in devices_list:
        try:
            print(device)
            benign_path = os.path.join(folder, device, benign_file)
            mirai_path = os.path.join(folder, device, m_folder)
            gafgyt_path = os.path.join(folder, device, g_folder)

            data_benign = pd.read_csv(benign_path)
            data_benign['label'] = 'Benign'
            device_data = data_benign

            if (device != "Ennino_Doorbell" and device != "Samsung_SNH"):
                for file in os.listdir(mirai_path):
                    file_name = os.path.join(mirai_path, file)
                    data = pd.read_csv(file_name)
                    label = file.partition('.')[0]
                    data['label'] = label
                    device_data = pd.concat([data, device_data], ignore_index=True)

            for file in os.listdir(gafgyt_path):
                file_name = os.path.join(gafgyt_path, file)
                data = pd.read_csv(file_name)
                label = file.partition('.')[0]
                data['label'] = label
                device_data = pd.concat([data, device_data], ignore_index=True)

            device_data = device_data.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
            device_data.dropna()
            if (device == "Damnini_Doorbell"):
                all_data = device_data
            else:
                all_data = pd.concat([all_data, device_data], ignore_index=True)

        except FileNotFoundError:
            print("Unable to find the file for: " + device)
            print("Check if the dataset is complete and try again")

    return all_data

# Feature Selection
# Used to eliminate the features with the least impact on the accuracy
def drop_some_columns(some_data):
    column_names = some_data.keys()

    for column in column_names:
        if ("pcc" in column or "radius" in column or "covariance" in column):
            some_data.drop([column], axis=1, inplace=True)
    return some_data


# Eliminates the features with the least impact on the accuracy
# Splits the data in training_data(X_train), testing_data(X_test), training_labels(Y_train), testing_labels(Y_test)
def split_data(all_data):
    all_data = drop_some_columns(all_data)
    labels = all_data['label']
    all_data = all_data.drop(['label'], axis=1)
    training_data, testing_data, training_labels, testing_labels = train_test_split(all_data, labels, test_size=0.3, random_state=12)
    return training_data, testing_data, training_labels, testing_labels

# def feature_importance(all_data):
#     start = time.time()

#     X = all_data.drop(['label', 'type'], axis=1)
#     Y = all_data['type']
#     %matplotlib inline
#     fig=plt.figure(figsize=(20, 40))
#     importances = mutual_info_classif(X,Y)
#     feat_importances = pd.Series(importances, X.columns[0:len(X.columns)])
#     feat_importances.plot(kind='barh')
#     plt.show()

#     end = time.time()
#     print("Time elapsed:", (end - start), "seconds", "\n")


# Classifiers

# KNN

# Creates and trains the classifier for a given device
def knn_classifier(training_data, training_labels, device):
    folder = "devices"

    print("Number of neighbors: " + str(3))
    knn = KNeighborsClassifier(3)
    knn.fit(training_data, training_labels)
    clasif_name = 'knn_clasif.sav'
    filename = os.path.join(folder, device, clasif_name)
    pickle.dump(knn, open(filename, 'wb'))


# Loads the classifier for KNN, given the data
# Creates the predictions
# Calculates, analyses and displays results
def knn_predict(testing_data, testing_labels, device, window):
    add_new_ti(window, "KNN")

    folder = "devices"
    clasif_name = 'knn_clasif.sav'
    filename = os.path.join(folder, device, clasif_name)

    start = time.time()
    knn = pickle.load(open(filename, 'rb'))
    predicted_labels = knn.predict(testing_data)

    test_conf_matrix = confusion_matrix(testing_labels, predicted_labels)
    print("Confusion matrix for Test Set:")
    print(test_conf_matrix)

    FP = test_conf_matrix.sum(axis=0) - np.diag(test_conf_matrix)
    FN = test_conf_matrix.sum(axis=1) - np.diag(test_conf_matrix)
    TP = np.diag(test_conf_matrix)

    print("False positives: " + str(FP.sum()) + " out of " + str(test_conf_matrix.sum()))
    add_new(window, "False positives: " + str(FP.sum()) + " out of " + str(test_conf_matrix.sum()))

    print(classification_report(testing_labels, predicted_labels))


    test_s = np.trace(test_conf_matrix) / test_conf_matrix.sum()
    test_scores = test_s

    end = time.time()
    print("Time elapsed: " + str(end - start) + "seconds" + "\n")
    add_new(window, "Time elapsed: " + str(end - start) + "seconds")


    print("Accuracy is: ", test_scores*100, "%")
    add_new(window, "Accuracy is: " + str(test_scores*100) + "%")



# Decision Tree Classifiers

# Creates and trains the classifier for a given device
def DT_gini_classifier(training_data, training_labels, device):
    folder = "devices"
    clasif_name = 'DT_gini_clasif.sav'
    filename = os.path.join(folder, device, clasif_name)

    dtc = DecisionTreeClassifier(criterion='gini', random_state=12, max_depth=None)

    dtc.fit(training_data, training_labels)
    pickle.dump(dtc, open(filename, 'wb'))



# Loads the classifier for DT_gini, given the data
# Creates the predictions
# Calculates, analyses and displays results
def DT_gini_predict(testing_data, testing_labels, device, window):
    add_new_ti(window, "DT_gini")


    folder = "devices"
    clasif_name = 'DT_gini_clasif.sav'
    filename = os.path.join(folder, device, clasif_name)

    start = time.time()
    DT_gini = pickle.load(open(filename, 'rb'))
    predicted_labels = DT_gini.predict(testing_data)

    test_conf_matrix = confusion_matrix(testing_labels, predicted_labels)
    print("Confusion matrix for Test Set:")
    print(test_conf_matrix)

    FP = test_conf_matrix.sum(axis=0) - np.diag(test_conf_matrix)
    FN = test_conf_matrix.sum(axis=1) - np.diag(test_conf_matrix)
    TP = np.diag(test_conf_matrix)

    print("False positives: " + str(FP.sum()) + " out of " + str(test_conf_matrix.sum()))
    add_new(window, "False positives: " + str(FP.sum()) + " out of " + str(test_conf_matrix.sum()))

    print(classification_report(testing_labels, predicted_labels))

    test_s = np.trace(test_conf_matrix) / test_conf_matrix.sum()
    test_scores = test_s

    end = time.time()
    print("Time elapsed: " + str(end - start) + "seconds" + "\n")
    add_new(window, "Time elapsed: " + str(end - start) + "seconds")


    print("Accuracy is: ", test_scores*100, "%")
    add_new(window, "Accuracy is: " + str(test_scores*100) + "%")


# Creates and trains the classifier for a given device
def DT_entropy_classifier(training_data, training_labels, device):
    folder = "devices"
    clasif_name = 'DT_entropy_clasif.sav'
    filename = os.path.join(folder, device, clasif_name)

    dtc = DecisionTreeClassifier(criterion='entropy', random_state=12, max_depth=None)

    dtc.fit(training_data, training_labels)
    pickle.dump(dtc, open(filename, 'wb'))


# Loads the classifier for DT_entropy, given the data
# Creates the predictions
# Calculates, analyses and displays results
def DT_entropy_predict(testing_data, testing_labels, device, window):
    add_new_ti(window, "DT_entropy")


    folder = "devices"
    clasif_name = 'DT_entropy_clasif.sav'
    filename = os.path.join(folder, device, clasif_name)

    start = time.time()
    DT_entropy = pickle.load(open(filename, 'rb'))
    predicted_labels = DT_entropy.predict(testing_data)

    test_conf_matrix = confusion_matrix(testing_labels, predicted_labels)
    print("Confusion matrix for Test Set:")
    print(test_conf_matrix)

    FP = test_conf_matrix.sum(axis=0) - np.diag(test_conf_matrix)
    FN = test_conf_matrix.sum(axis=1) - np.diag(test_conf_matrix)
    TP = np.diag(test_conf_matrix)

    print("False positives: " + str(FP.sum()) + " out of " + str(test_conf_matrix.sum()))
    add_new(window, "False positives: " + str(FP.sum()) + " out of " + str(test_conf_matrix.sum()))

    print(classification_report(testing_labels, predicted_labels))

    test_s = np.trace(test_conf_matrix) / test_conf_matrix.sum()
    test_scores = test_s

    end = time.time()
    print("Time elapsed: " + str(end - start) + "seconds" + "\n")
    add_new(window, "Time elapsed: " + str(end - start) + "seconds")


    print("Accuracy is: ", test_scores*100, "%")
    add_new(window, "Accuracy is: " + str(test_scores*100) + "%")



# Random Forest Classifiers

# Creates and trains the classifier for a given device
def RF_gini_classifier(training_data, training_labels, device):
    folder = "devices"
    clasif_name = 'RF_gini_clasif.sav'
    filename = os.path.join(folder, device, clasif_name)

    rfc = RandomForestClassifier(criterion='gini', random_state=12, max_depth=None)

    rfc.fit(training_data, training_labels)
    pickle.dump(rfc, open(filename, 'wb'))


# Loads the classifier for RF_gini, given the data
# Creates the predictions
# Calculates, analyses and displays results
def RF_gini_predict(testing_data, testing_labels, device, window):
    add_new_ti(window, "RF_gini")


    folder = "devices"
    clasif_name = 'RF_gini_clasif.sav'
    filename = os.path.join(folder, device, clasif_name)

    start = time.time()
    RF_gini = pickle.load(open(filename, 'rb'))
    predicted_labels = RF_gini.predict(testing_data)

    test_conf_matrix = confusion_matrix(testing_labels, predicted_labels)
    print("Confusion matrix for Test Set:")
    print(test_conf_matrix)

    FP = test_conf_matrix.sum(axis=0) - np.diag(test_conf_matrix)
    FN = test_conf_matrix.sum(axis=1) - np.diag(test_conf_matrix)
    TP = np.diag(test_conf_matrix)

    print("False positives: " + str(FP.sum()) + " out of " + str(test_conf_matrix.sum()))
    add_new(window, "False positives: " + str(FP.sum()) + " out of " + str(test_conf_matrix.sum()))

    print(classification_report(testing_labels, predicted_labels))

    test_s = np.trace(test_conf_matrix) / test_conf_matrix.sum()
    test_scores = test_s

    end = time.time()
    print("Time elapsed: " + str(end - start) + "seconds" + "\n")
    add_new(window, "Time elapsed: " + str(end - start) + "seconds")


    print("Accuracy is: ", test_scores*100, "%")
    add_new(window, "Accuracy is: " + str(test_scores*100) + "%")


# Creates and trains the classifier for a given device
def RF_entropy_classifier(training_data, training_labels, device):
    folder = "devices"
    clasif_name = 'RF_entropy_clasif.sav'
    filename = os.path.join(folder, device, clasif_name)

    rfc = RandomForestClassifier(criterion='entropy', random_state=12, max_depth=None)

    rfc.fit(training_data, training_labels)
    pickle.dump(rfc, open(filename, 'wb'))


# Loads the classifier for RF_entropy, given the data
# Creates the predictions
# Calculates, analyses and displays results
def RF_entropy_predict(testing_data, testing_labels, device, window):
    add_new_ti(window, "RF_entropy")

    folder = "devices"
    clasif_name = 'RF_entropy_clasif.sav'
    filename = os.path.join(folder, device, clasif_name)

    start = time.time()
    RF_entropy = pickle.load(open(filename, 'rb'))
    predicted_labels = RF_entropy.predict(testing_data)

    test_conf_matrix = confusion_matrix(testing_labels, predicted_labels)
    print("Confusion matrix for Test Set:")
    print(test_conf_matrix)

    FP = test_conf_matrix.sum(axis=0) - np.diag(test_conf_matrix)
    FN = test_conf_matrix.sum(axis=1) - np.diag(test_conf_matrix)
    TP = np.diag(test_conf_matrix)

    print("False positives: " + str(FP.sum()) + " out of " + str(test_conf_matrix.sum()))
    add_new(window, "False positives: " + str(FP.sum()) + " out of " + str(test_conf_matrix.sum()))

    print(classification_report(testing_labels, predicted_labels))

    test_s = np.trace(test_conf_matrix) / test_conf_matrix.sum()
    # test_s = (TP+TN)/(TP+FP+FN+TN)
    test_scores = test_s

    end = time.time()
    print("Time elapsed: " + str(end - start) + "seconds" + "\n")
    add_new(window, "Time elapsed: " + str(end - start) + "seconds")


    print("Accuracy is: ", test_scores*100, "%")
    add_new(window, "Accuracy is: " + str(test_scores*100) + "%")


def SVM_rbf_classifier(training_data, training_labels, device):
    folder = "devices"

    svm = SVC(kernel='rbf', random_state=12)
    svm.fit(training_data, training_labels)
    clasif_name = 'SVM_rbf_clasif.sav'
    filename = os.path.join(folder, device, clasif_name)
    pickle.dump(svm, open(filename, 'wb'))


# Loads the classifier for SVM_RBF, given the data
# Creates the predictions
# Calculates, analyses and displays results
def SVM_rbf_predict(testing_data, testing_labels, device, window):
    add_new_ti(window, "SVM_RBF")

    folder = "devices"
    clasif_name = 'SVM_rbf_clasif.sav'
    filename = os.path.join(folder, device, clasif_name)

    start = time.time()
    svm = pickle.load(open(filename, 'rb'))
    predicted_labels = svm.predict(testing_data)

    test_conf_matrix = confusion_matrix(testing_labels, predicted_labels)
    print("Confusion matrix for Test Set:")
    print(test_conf_matrix)

    FP = test_conf_matrix.sum(axis=0) - np.diag(test_conf_matrix)
    FN = test_conf_matrix.sum(axis=1) - np.diag(test_conf_matrix)
    TP = np.diag(test_conf_matrix)

    print("False positives: " + str(FP.sum()) + " out of " + str(test_conf_matrix.sum()))
    add_new(window, "False positives: " + str(FP.sum()) + " out of " + str(test_conf_matrix.sum()))

    print(classification_report(testing_labels, predicted_labels))


    test_s = np.trace(test_conf_matrix) / test_conf_matrix.sum()
    test_scores = test_s

    end = time.time()
    print("Time elapsed: " + str(end - start) + "seconds" + "\n")
    add_new(window, "Time elapsed: " + str(end - start) + "seconds")


    print("Accuracy is: ", test_scores*100, "%")
    add_new(window, "Accuracy is: " + str(test_scores*100) + "%")


def SVM_linear_classifier(training_data, training_labels, device):
    folder = "devices"

    svm = SVC(kernel='linear', random_state=12)
    svm.fit(training_data, training_labels)
    clasif_name = 'SVM_linear_clasif.sav'
    filename = os.path.join(folder, device, clasif_name)
    pickle.dump(svm, open(filename, 'wb'))


# Loads the classifier for SVM_Linear, given the data
# Creates the predictions
# Calculates, analyses and displays results
def SVM_linear_predict(testing_data, testing_labels, device, window):
    add_new_ti(window, "SVM_Linear")

    folder = "devices"
    clasif_name = 'SVM_linear_clasif.sav'
    filename = os.path.join(folder, device, clasif_name)

    start = time.time()
    svm = pickle.load(open(filename, 'rb'))
    predicted_labels = svm.predict(testing_data)

    test_conf_matrix = confusion_matrix(testing_labels, predicted_labels)
    print("Confusion matrix for Test Set:")
    print(test_conf_matrix)

    FP = test_conf_matrix.sum(axis=0) - np.diag(test_conf_matrix)
    FN = test_conf_matrix.sum(axis=1) - np.diag(test_conf_matrix)
    TP = np.diag(test_conf_matrix)

    print("False positives: " + str(FP.sum()) + " out of " + str(test_conf_matrix.sum()))
    add_new(window, "False positives: " + str(FP.sum()) + " out of " + str(test_conf_matrix.sum()))

    print(classification_report(testing_labels, predicted_labels))

    test_s = np.trace(test_conf_matrix) / test_conf_matrix.sum()
    test_scores = test_s

    end = time.time()
    print("Time elapsed: " + str(end - start) + "seconds" + "\n")
    add_new(window, "Time elapsed: " + str(end - start) + "seconds")

    print("Accuracy is: ", test_scores * 100, "%")
    add_new(window, "Accuracy is: " + str(test_scores * 100) + "%")


# Loads a classifier for a certain device
def load_clasif(device, clasif, testing_data, testing_labels):
    folder = "devices"
    if (clasif == 'knn'):
        clasif_name = 'knn_clasif.sav'
    elif (clasif == "DT_gini"):
        clasif_name = 'DT_gini_clasif.sav'
    elif (clasif == "DT_entropy"):
        clasif_name = 'DT_entropy_clasif.sav'
    elif (clasif == "RF_gini"):
        clasif_name = 'RF_gini_clasif.sav'
    elif (clasif == "RF_entropy"):
        clasif_name = 'RF_entropy_clasif.sav'
    else:
        clasif_name = None
    filename = os.path.join(folder, device, clasif_name)

    start = time.time()
    classifier = pickle.load(open(filename, 'rb'))
