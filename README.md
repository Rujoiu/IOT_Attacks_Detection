# IOT Attacks Detection
Created by Alexandru Rujoiu-Mare, March 2022

<h3> 1. What should each folder contain </h3>

After downloading the files and unzipping, you should have a folder that looks like this:

- devices (folder)
- functions.py
- icon.png
- icon1.ico
- main.py
- nn.py
- README.md
- requirements.txt
- window_functions.py

The "devices" folder contains 10 folders: 
- all
- Damnini_Doorbell
- Ecobee_Thermostat
- Ennino_Doorbell
- Philips_B120N10_Baby_Monitor
- Provision_PT_737E_Security_Camera
- Provision_PT_838_Security_Camera
- Samsung_SNH
- SimpleHome_1002_SecurityCamera
- SimpleHome_1003_SecurityCamera 

Each of these folders (except from "all") contains:
- benign_traffic.csv 
- gafgyt_attacks (that contains combo.cvs, junk.csv, scan.csv, tcp.csv, udp.csv) 
- mirai_attacks (that contains ack.cvs, scan.csv, syn.csv, udp.csv, udpplain.csv) - Ennio_Doorbell and Samsung_SNH do not contain mirai_attacks
- and the already trained on the data of the specific device (DT_entropy_clasif.sav, DT_gini_clasif.sav, KNN_classif.sav, RF_entropy_clasif.sav, RF_gini_clasif.sav)

The "all" folder contains the classifiers trained on all the dataset.

Before running main.py, you need to install the libraries from requirements.txt.

If you want to run the neural network on the GPU as well, you will also need to install the latest versions of CUDA and CUDNN.

If not, the NNs will not run on the GPU and you will get the warnings:

"W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found"

"I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine."

and you will also get some warnings when running the Neural Network, saying that it could not load another dynamic library.

This does not affect in any way the accuracy or the way the algorthm works, but it does affect the time it takes to train the neural networks.


<h3>2. How to use the GUI </h3>

There are 5 buttons:
- The Info button (top right corner)
- Train Classifier
- Select Device
- Select Classifier
- Neural Networks

**Train Classifier**

Allows you to train or re-train the classifiers trained on the data from each device and the classifiers trained on all the dataset ("All" Classifier).

If you have downloaded the already existent classifiers, you should not have to use this button, but if you did not, you have to train the classifiers first.


**Select Device**

Here you can test the 5 classifiers already trained on each device.

IMPORTANT: Each device will use the classifiers train on its own data.

"Run All" will test each device sequentially, following the alphabetical order.


**Select Classifier**

Here you are able to select first what classifiers you want to use (train on whose device's data) and on what data you want to test them.

For example, you can test the classifiers trained on the data from Damnini_Doorbell on the data from Ecobee_Thermostat.

You are also able to use the classifiers train on the data from all the devices. To do that, just select " "ALL" classifier "

As Ennino Doorbell and Samsung SNH do not have Mirai Attacks, I would recommend not selecting their classifiers to run on other devices' data. You are able to do this, but the accuracy will be very low, for obvious reasons.

NOTE: When testing the data for all the devices, data is loaded sequentially, one device at the time, and concatenated. Please be patient.

**For these 3 buttons, each command will open a top level window where details about the progress and the results will be displayed** 

**Neural Networks**

This button opens the window of selection for the neural network that you want to use.

There are 4 options, each with 4 numbers of epochs for training.

RELU - Rectified Linear Unit.

TAHN - Hyperbolic Tangent.

SCALED Data - the values have been scaled before training (the values can be greater than 1).

NORMALIZED Data - the values have been normalized before training (the values are between 0-1).

As Keras library does not allow (the progress bar being auto-generated), the results and the progress of the NNs will not be displayed in a top level window. After selecting the neural network that you want to use, you have go back to the terminal. There you can see the summary of the NN and the progress, as well as the accuracy on both train and test set.

<h3>3. Other notes </h3>

Expect KNN to train fast, but to take a longer time to predict (also expect this classifier to have a much bigger size than the others).

Decision Tree and Random Forest classifiers make prediction fast, but they take a longer time to train.

You can also find the code here (but without pre-trained classifiers nor the data set): https://github.com/Rujoiu/IOT_Attacks_Detection

Data Set: https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT





