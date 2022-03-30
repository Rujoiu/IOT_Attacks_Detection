import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import functions as fc


def get_data(device):
    print("Loading Data")
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
    try:
        if device != "Ennio_Doorbell" and device != "Samsung_SNH":
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
    except FileNotFoundError:
        print("Unable to find the file for: " + device)
        print("Check if the dataset is complete and try again")

    device_data = device_data.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    device_data.dropna()

    labelencoder = LabelEncoder()
    print("Data Loaded")

    device_data['label'] = labelencoder.fit_transform(device_data['label'])
    labels = device_data['label']
    all_data = device_data.drop(['label'], axis=1)

    array_data = all_data.to_numpy()
    array_labels = labels.to_numpy()

    del all_data
    del device_data

    training_data, testing_data, training_labels, testing_labels = train_test_split(array_data, array_labels, test_size=0.3, random_state=12)

    scaler = StandardScaler().fit(training_data)
    training_data = scaler.transform(training_data)
    testing_data = scaler.transform(testing_data)
    return training_data, training_labels, testing_data, testing_labels


def get_data_normalized(device):
    print("Loading Data")
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
    try:
        if device != "Ennio_Doorbell" and device != "Samsung_SNH":
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
    except FileNotFoundError:
        print("Unable to find the file for: " + device)
        print("Check if the dataset is complete and try again")

    device_data = device_data.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    device_data.dropna()

    labelencoder = LabelEncoder()
    print("Data Loaded")

    device_data['label'] = labelencoder.fit_transform(device_data['label'])
    labels = device_data['label']
    all_data = device_data.drop(['label'], axis=1)

    array_data = all_data.to_numpy()
    array_labels = labels.to_numpy()

    del all_data
    del device_data

    training_data, testing_data, training_labels, testing_labels = train_test_split(array_data, array_labels, test_size=0.3, random_state=12)

    scaler = StandardScaler().fit(training_data)
    training_data = scaler.transform(training_data)
    testing_data = scaler.transform(testing_data)

    training_data = tf.keras.utils.normalize(training_data, axis=1)
    testing_data = tf.keras.utils.normalize(testing_data, axis=1)

    return training_data, training_labels, testing_data, testing_labels


def get_data_all():
    print("Loading Data")
    all_data = fc.new_functia_desteapta_all()
    labelencoder = LabelEncoder()
    print("Data Loaded")

    all_data['label'] = labelencoder.fit_transform(all_data['label'])
    labels = all_data['label']
    all_data = all_data.drop(['label'], axis=1)

    array_data = all_data.to_numpy()
    array_labels = labels.to_numpy()

    del all_data

    training_data, testing_data, training_labels, testing_labels = train_test_split(array_data, array_labels,
                                                                                    test_size=0.3, random_state=12)

    scaler = StandardScaler().fit(training_data)
    training_data = scaler.transform(training_data)
    testing_data = scaler.transform(testing_data)
    return training_data, training_labels, testing_data, testing_labels


def get_data_all_normalized():
    print("Loading Data")
    all_data = fc.new_functia_desteapta_all()
    labelencoder = LabelEncoder()
    print("Data Loaded")

    all_data['label'] = labelencoder.fit_transform(all_data['label'])
    labels = all_data['label']
    all_data = all_data.drop(['label'], axis=1)

    array_data = all_data.to_numpy()
    array_labels = labels.to_numpy()

    del all_data

    training_data, testing_data, training_labels, testing_labels = train_test_split(array_data, array_labels,
                                                                                    test_size=0.3, random_state=12)

    scaler = StandardScaler().fit(training_data)
    training_data = scaler.transform(training_data)
    testing_data = scaler.transform(testing_data)

    training_data = tf.keras.utils.normalize(training_data, axis=1)
    testing_data = tf.keras.utils.normalize(testing_data, axis=1)

    return training_data, training_labels, testing_data, testing_labels


def create_model_relu(device, epochs_no):
    if device != "all":
        training_data, training_labels, testing_data, testing_labels = get_data(device)
    else:
        training_data, training_labels, testing_data, testing_labels = get_data_all()

    model = Sequential()
    model.add(Dense(700, input_dim = 115, activation='relu'))
    model.add(Dense(600, activation='relu'))
    model.add(Dense(450, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(11, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    print(model.summary())
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(training_data, training_labels, epochs=epochs_no, validation_data=(testing_data, testing_labels))


def create_model_relu_normalized(device, epochs_no):
    if device != "all":
        training_data, training_labels, testing_data, testing_labels = get_data_normalized(device)
    else:
        training_data, training_labels, testing_data, testing_labels = get_data_all_normalized()

    model = Sequential()
    model.add(Dense(700, input_dim = 115, activation='relu'))
    model.add(Dense(600, activation='relu'))
    model.add(Dense(450, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(11, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    print(model.summary())
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.fit(training_data, training_labels, epochs=epochs_no, validation_data=(testing_data, testing_labels))


def create_model_tanh(device, epochs_no):
    if device != "all":
        training_data, training_labels, testing_data, testing_labels = get_data(device)
    else:
        training_data, training_labels, testing_data, testing_labels = get_data_all()

    input_dim = 115

    model2 = Sequential()
    model2.add(Dense(int(0.75 * input_dim), activation="tanh", input_shape=(input_dim,)))
    model2.add(Dense(int(0.5 * input_dim), activation="tanh"))
    model2.add(Dense(int(0.33 * input_dim), activation="tanh"))
    model2.add(Dense(int(0.25 * input_dim), activation="tanh"))
    model2.add(Dense(int(0.33 * input_dim), activation="tanh"))
    model2.add(Dense(int(0.5 * input_dim), activation="tanh"))
    model2.add(Dense(int(0.75 * input_dim), activation="tanh"))
    model2.add(Dense(input_dim))
    model2.add(Dense(11, activation='softmax'))
    print(model2.summary())
    model2.compile(loss='sparse_categorical_crossentropy',
                        optimizer="adam",
                        metrics=['accuracy'])
    model2.fit(training_data, training_labels, epochs=epochs_no, validation_data=(testing_data, testing_labels))


def create_model_tanh_normalized(device, epochs_no):
    if device != "all":
        training_data, training_labels, testing_data, testing_labels = get_data_normalized(device)
    else:
        training_data, training_labels, testing_data, testing_labels = get_data_all_normalized()

    input_dim = 115

    model2 = Sequential()
    model2.add(Dense(int(0.75 * input_dim), activation="tanh", input_shape=(input_dim,)))
    model2.add(Dense(int(0.5 * input_dim), activation="tanh"))
    model2.add(Dense(int(0.33 * input_dim), activation="tanh"))
    model2.add(Dense(int(0.25 * input_dim), activation="tanh"))
    model2.add(Dense(int(0.33 * input_dim), activation="tanh"))
    model2.add(Dense(int(0.5 * input_dim), activation="tanh"))
    model2.add(Dense(int(0.75 * input_dim), activation="tanh"))
    model2.add(Dense(input_dim))
    model2.add(Dense(11, activation='softmax'))
    print(model2.summary())
    model2.compile(loss='sparse_categorical_crossentropy',
                        optimizer="adam",
                        metrics=['accuracy'])
    model2.fit(training_data, training_labels, epochs=epochs_no, validation_data=(testing_data, testing_labels))