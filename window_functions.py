import functions as fc
import tkinter as tk


# The following 4 methods are used to write in the top level window
def add_new(window, text):
    window.add_new_text(text)


def add_new_ti(window, text):
    window.add_new_title(text)


def add_new_de(window, text):
    window.add_new_device(text)


def add_new_err(window, text):
    window.add_new_error(text)


def print_device(device, window):
    print("=" * 70)
    print('\033[4m' + "\nDevice: " + device)
    add_new_de(window, "Device: " + device)


# Trains KNN, DT and RF classifiers for a given device
def retrain(device, window):
    print_device(device, window)
    print('\033[0m' + "\nLoading data")
    add_new(window, "Loading data")
    if (device != "all"):
        try:
            all_data = fc.functia_desteapta(device)
        except FileNotFoundError:
            print("Unable to find the file for: " + device)
            print("Check if the dataset is complete and try again")
            add_new_err(window, "Unable to find the file for: " + device)
            add_new_err(window, "Check if the dataset is complete and try again")
    else:
        all_data = fc.functia_desteapta_all(window)
    training_data, testing_data, training_labels, testing_labels = fc.split_data(all_data)
    print("\nData Loaded")
    add_new(window, "Data Loaded")
    fc.DT_gini_classifier(training_data, training_labels, device)
    print("DT_gini Classifier Created")
    add_new_ti(window, "DT_gini Classifier Created")
    fc.DT_entropy_classifier(training_data, training_labels, device)
    print("DT_entropy Classifier Created")
    add_new_ti(window, "DT_entropy Classifier Created")
    fc.RF_gini_classifier(training_data, training_labels, device)
    print("RF_gini Classifier Created")
    add_new_ti(window, "RF_gini Classifier Created")
    fc.RF_entropy_classifier(training_data, training_labels, device)
    print("RF_entropy Classifier Created")
    add_new_ti(window, "RF_entropy Classifier Created")
    fc.knn_classifier(training_data, training_labels, device)
    print("KNN Classifier Created")
    add_new_ti(window, "KNN Classifier Created")
    # fc.SVM_rbf_classifier(training_data, training_labels, device)
    # print("SVM_RBF Classifier Created")
    # add_new_ti(window, "SVM_RBF Classifier Created")
    # fc.SVM_linear_classifier(training_data, training_labels, device)
    # print("SVM_Linear Classifier Created")
    # add_new_ti(window, "SVM_Linear Classifier Created")
    print("=" * 70)


# Tests the already trained KNN, DT and RF classifiers from a device on the testing data from the same device
def select_device(device, window):
    print_device(device, window)
    print('\033[0m' + "\nLoading data")
    add_new(window, "Loading data")
    if device != "all":
        try:
            all_data = fc.functia_desteapta(device)
        except FileNotFoundError:
            print("Unable to find the file for: " + device)
            print("Check if the dataset is complete and try again")
            add_new_err(window, "Unable to find the file for: " + device)
            add_new_err(window, "Check if the dataset is complete and try again")
        training_data, testing_data, training_labels, testing_labels = fc.split_data(all_data)
        print("\nData Loaded")
        add_new(window, "Data Loaded")
        try:
            print("\nDT_gini")
            fc.DT_gini_predict(testing_data, testing_labels, device, window)
            print("\nDT_entropy")
            fc.DT_entropy_predict(testing_data, testing_labels, device, window)
            print("\nRF_gini")
            fc.RF_gini_predict(testing_data, testing_labels, device, window)
            print("\nRF_entropy")
            fc.RF_entropy_predict(testing_data, testing_labels, device, window)
            print("\nKNN")
            fc.knn_predict(testing_data, testing_labels, device, window)
        except FileNotFoundError:
            print("Unable to find the classifier")
            print("Train the classifier and try again")
            add_new_err(window, "Unable to find the classifier")
            add_new_err(window, "Train the classifier and try again")
        print("=" * 70)
    else:
        devices_list = ["Damnini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell", "Philips_B120N10_Baby_Monitor",
                        "Provision_PT_737E_Security_Camera", "Provision_PT_838_Security_Camera", "Samsung_SNH",
                        "SimpleHome_1002_SecurityCamera", "SimpleHome_1003_SecurityCamera"]
        for dev in devices_list:
            select_device(dev, window)


# Tests the already trained KNN, DT and RF classifiers from a device on the testing data from any device
# "device" - the device from which the data will be loaded
# "classif" - the device from which the classifiers will be loaded
def select_data(device, classif, window):
    print("=" * 70)
    print('\033[4m' + "\nClassifiers from: " + classif)
    add_new_de(window, "Classifiers from: " + str(classif))
    print('\033[4m' + "\nData from: " + device)
    add_new_de(window, "Data from: " + str(device))
    print('\033[0m' + "\nLoading data")
    add_new(window, "Loading data for: " + device)
    if device != "all":
        try:
            all_data = fc.functia_desteapta(device)
        except(FileNotFoundError):
            print("Unable to find the file for: " + device)
            print("Check if the dataset is complete and try again")
            add_new_err(window, "Unable to find the file for: " + device)
            add_new_err(window, "Check if the dataset is complete and try again")
    else:
        all_data = fc.functia_desteapta_all(window)
    training_data, testing_data, training_labels, testing_labels = fc.split_data(all_data)
    print("Data Loaded")
    add_new(window, "Data Loaded")

    try:
        print("\nDT_gini")
        fc.DT_gini_predict(testing_data, testing_labels, classif, window)
        print("\nDT_entropy")
        fc.DT_entropy_predict(testing_data, testing_labels, classif, window)
        print("\nRF_gini")
        fc.RF_gini_predict(testing_data, testing_labels, classif, window)
        print("\nRF_entropy")
        fc.RF_entropy_predict(testing_data, testing_labels, classif, window)
        print("\nKNN")
        fc.knn_predict(testing_data, testing_labels, classif, window)
    except(FileNotFoundError):
        print("Unable to find the classifier")
        print("Train the classifier and try again")
        add_new_err(window, "Unable to find the classifier")
        add_new_err(window, "Train the classifier and try again")
    print("=" * 70)