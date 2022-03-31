import dataclasses
import tkinter as tk
import window_functions as wf
import nn as nn


small_times_font = "Times 20 bold italic"
text_font = "Times 14"
device_data = ""


@dataclasses.dataclass
class state:
    classif: str


# Top Level Window Class used to print the results
# At first it contains Run and Back buttons and scrollbar
# The class contains methods to add labels with different fonts
class Window(tk.Toplevel):
    def __init__(self, parent, controller, function, device):
        super().__init__(parent)
        self.controller = controller
        self.par = parent

        self.geometry('600x650')
        self.title('Toplevel Window')

        self.aframe = tk.Frame(self)
        self.aframe.pack(fill=tk.BOTH, expand=1)

        self.my_canvas = tk.Canvas(self.aframe, bg="#ADD8E6")
        self.my_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.sc_bar = tk.Scrollbar(self.aframe, orient=tk.VERTICAL, command=self.my_canvas.yview)
        self.sc_bar.pack(side=tk.RIGHT, fill=tk.Y)

        self.my_canvas.configure(yscrollcommand=self.sc_bar.set)
        self.my_canvas.bind('<Configure>', lambda e: self.my_canvas.configure(scrollregion=self.my_canvas.bbox("all")))

        self.sec_frame = tk.Frame(self.my_canvas)
        self.sec_frame.grid(row=0, column=0, sticky="nsew")
        self.sec_frame.configure(width="600", height="650", bg="#ADD8E6")
        self.sec_frame.columnconfigure(3, weight=2)
        self.my_canvas.create_window(0, 0, window=self.sec_frame, anchor=tk.NW)

        if function == wf.select_device:
            btn = tk.Button(self, text='Run',
                            command=lambda: wf.select_device(device, self),
                            height=2, width=15, bg="#32CD32")
            btn.place(relx=0.85, rely=0.2, anchor=tk.CENTER)
        elif function == wf.select_data:
            btn = tk.Button(self, text='Run',
                            command=lambda: wf.select_data(device, controller.state.classif, self),
                            height=2, width=15, bg="#32CD32")
            btn.place(relx=0.85, rely=0.2, anchor=tk.CENTER)
        elif function == wf.retrain:
            btn = tk.Button(self, text='Run',
                            command=lambda: wf.retrain(device, self),
                            height=2, width=15, bg="#32CD32")
            btn.place(relx=0.85, rely=0.2, anchor=tk.CENTER)


        close = tk.Button(self, text='Close',
                          command=self.destroy,
                          height=2, width=15, bg="#FF0000")
        close.place(relx=0.85, rely=0.8, anchor=tk.CENTER)


    def add_new_text(self, text):
        var = text
        vars()[var] = tk.Label(self.sec_frame, text=text, background="#ADD8E6", font="Times 15" )
        vars()[var].pack(fill=tk.BOTH, expand=1)
        self.update()


    def add_new_title(self, text):
        var = text
        vars()[var] = tk.Label(self.sec_frame, text=text, background="#ADD8E6", font="Times 15 bold italic" )
        vars()[var].pack(fill=tk.BOTH, expand=1)
        self.update()


    def add_new_device(self, text):
        var = text
        vars()[var] = tk.Label(self.sec_frame, text=text, background="#ADD8E6", font="Times 17 underline bold italic" )
        vars()[var].pack(fill=tk.BOTH, expand=1)
        self.update()

    def add_new_error(self, text):
        var = text
        vars()[var] = tk.Label(self.sec_frame, text=text, background="#ADD8E6", font="Times 17 underline bold italic", fg="red")
        vars()[var].pack(fill=tk.BOTH, expand=1)
        self.update()


# The Root
# Initiates all the frames
# Creates the state object
# show_frame method is used to navigate through the frames
# open_window method is used to create a new top level window
class Detection(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self.state = state(classif="")

        for F in (HomeWindow, RetrainWindow, InfoWindow, SelectDeviceWindow, SelectDataWindow, SelectClasifWindow, NNWindow1, NNWindow2):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            frame.configure(width="600", height="650", background="#ADD8E6")

        self.show_frame(HomeWindow)

    def show_frame(self, cont):
        frame = self.frames[cont]
        self.state = state(classif="")
        frame.tkraise()

    def open_window(self, controller, function, device):
        window = Window(self, controller, function, device)
        window.grab_set()


# The home window, that contains the different choices the user has
class HomeWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        intro_label = tk.Label(self, text="Choose your action", background="#ADD8E6", font=small_times_font)
        intro_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # INFO Button
        b_info = tk.Button(self, text="i", height=1, width=2,
                           command=lambda: controller.show_frame(InfoWindow),
                           bg="#FFFF76")
        b_info.place(relx=0.97, rely=0.03, anchor=tk.CENTER)

        # Train Label and Button
        retrain_label = tk.Label(self, text="Re-train the Classifiers", background="#ADD8E6",
                                   font=small_times_font)
        retrain_label.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

        b_train_clasif = tk.Button(self, text='Train Classifiers', height=4, width=20,
                                   command=lambda: controller.show_frame(RetrainWindow),
                                   bg="#1776FF")
        b_train_clasif.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

        # Select device Labels and Buttons
        select_label = tk.Label(self, text="Use the already trained Classifiers", background="#ADD8E6",
                                  font=small_times_font)
        select_label.place(relx=0.5, rely=0.45, anchor=tk.CENTER)

        b_select_device = tk.Button(self, text='Select Device', height=4, width=20,
                                    command=lambda: controller.show_frame(SelectDeviceWindow),
                                    bg="#1776FF")
        b_select_clasif = tk.Button(self, text='Select Classifier', height=4, width=20,
                                    command=lambda: controller.show_frame(SelectClasifWindow),
                                    bg="#1776FF")
        b_select_device.place(relx=0.25, rely=0.55, anchor=tk.CENTER)
        b_select_clasif.place(relx=0.75, rely=0.55, anchor=tk.CENTER)

        # Neural Networks
        select_label = tk.Label(self, text="Neural Networks", background="#ADD8E6", font=small_times_font)
        select_label.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
        b_nn = tk.Button(self, text='Neural Networks', height=4, width=20,
                         command=lambda: controller.show_frame(NNWindow1))
        b_nn.place(relx=0.5, rely=0.8, anchor=tk.CENTER)


# The info window, that contains information the user should know
class InfoWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        info_label = tk.Label(self, text="Here is what you need to know:", background="#ADD8E6", font=small_times_font)
        info_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        some_text = tk.Label(self, text="The blue buttons are for KNN, DT and RF classifiers."
                                        "\nYou can retrain the classifiers for each device or use the already trained ones that I have created."
                                        " Select Device button will run the KNN, DT, RF classifiers "
                                        "of a device on the testing data from the same device.  "
                                        "\nSelect Classifier Button allows you to select the classifiers from"
                                        "a device first, then to run them on the testing data of a different device.\n "
                                        "\nIMPORTANT "
                                        "\nEnnino Doorbell and Samsung SNH do not have Mirai Attacks"
                                        "\nSo, I would recommend not selecting their classifiers to run on other devices' data. "
                                        "You are able to do this, but the accuracy will be very low, for obvious reasons.",
                             background="#ADD8E6", font=text_font, wraplength=550, justify=tk.CENTER)
        some_text.place(relx=0.5, rely=0.42, anchor=tk.CENTER)

        author_text = tk.Label(self, text="Created by: Alexandru Rujoiu-Mare \nUniversity of Manchester, 2022", background="#ADD8E6", font="Times 12 bold underline italic")
        author_text.place(relx=0.5, rely=0.80, anchor=tk.CENTER)

        info_back = tk.Button(self, text="Back", height=2, width=10,
                              command=lambda: controller.show_frame(HomeWindow),
                              bg="#FF0000")
        info_back.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


# The Retrain Window - You can choose the device whose classifiers you want to train/retrain
class RetrainWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Choose what device you want to train", background="#ADD8E6", font=small_times_font)
        label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        retrain_device1 = tk.Button(self, text="Damnini Doorbell",
                                    command=lambda: self.open_window(controller, wf.retrain, "Damnini_Doorbell"), height=3, width=20)
        retrain_device1.place(relx=0.3, rely=0.3, anchor=tk.CENTER)

        retrain_device2 = tk.Button(self, text="Ecobee Thermostat",
                                    command=lambda: self.open_window(controller, wf.retrain, "Ecobee_Thermostat"), height=3, width=20)
        retrain_device2.place(relx=0.3, rely=0.4, anchor=tk.CENTER)

        retrain_device3 = tk.Button(self, text="Ennio Doorbell",
                                    command=lambda: self.open_window(controller, wf.retrain, "Ennio_Doorbell"), height=3, width=20)
        retrain_device3.place(relx=0.3, rely=0.5, anchor=tk.CENTER)

        retrain_device4 = tk.Button(self, text="Philips Baby Monitor",
                                    command=lambda: self.open_window(controller, wf.retrain, "Philips_B120N10_Baby_Monitor"), height=3, width=20)
        retrain_device4.place(relx=0.3, rely=0.6, anchor=tk.CENTER)

        retrain_device5 = tk.Button(self, wraplength=100, text="Provision Security Camera - 737E",
                                    command=lambda: self.open_window(controller, wf.retrain, "Provision_PT_737E_Security_Camera"), height=3, width=20)
        retrain_device5.place(relx=0.3, rely=0.7, anchor=tk.CENTER)

        retrain_device6 = tk.Button(self, wraplength=100, text="Provision Security Camera - 838",
                                    command=lambda: self.open_window(controller, wf.retrain, "Provision_PT_838_Security_Camera"), height=3, width=20)
        retrain_device6.place(relx=0.7, rely=0.3, anchor=tk.CENTER)

        retrain_device7 = tk.Button(self, text="Samsung SNH",
                                    command=lambda: self.open_window(controller, wf.retrain, "Samsung_SNH"), height=3, width=20)
        retrain_device7.place(relx=0.7, rely=0.4, anchor=tk.CENTER)

        retrain_device8 = tk.Button(self, wraplength=100, text="SimpleHome Security Camera - 1002",
                                    command=lambda: self.open_window(controller, wf.retrain, "SimpleHome_1002_SecurityCamera"), height=3, width=20)
        retrain_device8.place(relx=0.7, rely=0.5, anchor=tk.CENTER)

        retrain_device9 = tk.Button(self, wraplength=100, text="SimpleHome Security Camera - 1003",
                                    command=lambda: self.open_window(controller, wf.retrain, "SimpleHome_1003_SecurityCamera"), height=3, width=20)
        retrain_device9.place(relx=0.7, rely=0.6, anchor=tk.CENTER)

        retrain_all = tk.Button(self, text="\"ALL\" classifier",
                                command=lambda: self.open_window(controller, wf.retrain, "all"), height=3, width=20)
        retrain_all.place(relx=0.7, rely=0.7, anchor=tk.CENTER)

        retrain_back = tk.Button(self, text="Back", height=2, width=10,
                                 command=lambda: controller.show_frame(HomeWindow),
                                 bg="#FF0000")
        retrain_back.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

    def open_window(self, controller, function, device):
        window = Window(self, controller, function, device)
        window.grab_set()


# Tests the already trained KNN, DT and RF classifiers from a device on the testing data from the same device
class SelectDeviceWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Select the device", background="#ADD8E6", font=small_times_font)
        label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        select_device_device1 = tk.Button(self, text="Damnini Doorbell",
                                          command=lambda: self.open_window(controller, wf.select_device, "Damnini_Doorbell"), height=3, width=20)
        select_device_device1.place(relx=0.3, rely=0.3, anchor=tk.CENTER)

        select_device_device2 = tk.Button(self, text="Ecobee Thermostat",
                                          command=lambda: self.open_window(controller, wf.select_device, "Ecobee_Thermostat"), height=3, width=20)
        select_device_device2.place(relx=0.3, rely=0.4, anchor=tk.CENTER)

        select_device_device3 = tk.Button(self, text="Ennio Doorbell",
                                          command=lambda: self.open_window(controller, wf.select_device, "Ennio_Doorbell"), height=3, width=20)
        select_device_device3.place(relx=0.3, rely=0.5, anchor=tk.CENTER)

        select_device_device4 = tk.Button(self, text="Philips Baby Monitor",
                                          command=lambda: self.open_window(controller, wf.select_device, "Philips_B120N10_Baby_Monitor"), height=3, width=20)
        select_device_device4.place(relx=0.3, rely=0.6, anchor=tk.CENTER)

        select_device_device5 = tk.Button(self, wraplength=100, text="Provision Security Camera - 737E",
                                          command=lambda: self.open_window(controller, wf.select_device, "Provision_PT_737E_Security_Camera"), height=3, width=20)
        select_device_device5.place(relx=0.3, rely=0.7, anchor=tk.CENTER)

        select_device_device6 = tk.Button(self, wraplength=100, text="Provision Security Camera - 838",
                                          command=lambda: self.open_window(controller, wf.select_device, "Provision_PT_838_Security_Camera"), height=3, width=20)
        select_device_device6.place(relx=0.7, rely=0.3, anchor=tk.CENTER)

        select_device_device7 = tk.Button(self, text="Samsung SNH",
                                          command=lambda: self.open_window(controller, wf.select_device, "Samsung_SNH"), height=3, width=20)
        select_device_device7.place(relx=0.7, rely=0.4, anchor=tk.CENTER)

        select_device_device8 = tk.Button(self, wraplength=100, text="SimpleHome Security Camera - 1002",
                                          command=lambda: self.open_window(controller, wf.select_device, "SimpleHome_1002_SecurityCamera"), height=3, width=20)
        select_device_device8.place(relx=0.7, rely=0.5, anchor=tk.CENTER)

        select_device_device9 = tk.Button(self, wraplength=100, text="SimpleHome Security Camera - 1003",
                                          command=lambda: self.open_window(controller, wf.select_device, "SimpleHome_1003_SecurityCamera"), height=3, width=20)
        select_device_device9.place(relx=0.7, rely=0.6, anchor=tk.CENTER)

        select_device_all = tk.Button(self, text="Run All",
                                      command=lambda: self.open_window(controller, wf.select_device, "all"), height=3, width=20)
        select_device_all.place(relx=0.7, rely=0.7, anchor=tk.CENTER)

        select_device_back = tk.Button(self, text="Back", height=2, width=10,
                                       command=lambda: controller.show_frame(HomeWindow),
                                       bg="#FF0000")
        select_device_back.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

    def open_window(self, controller, function, device):
        window = Window(self, controller, function, device)
        window.grab_set()


# Tests the already trained KNN, DT and RF classifiers from a device on the testing data from any device
# uses wf.select_data
# "device" - the device from which the data will be loaded
# "classif" - the device from which the classifiers will be loaded
# In this window you select the "classif" (the classifiers you want to use)
# "classif is passed into the next frame, using the controller.state object already created
class SelectClasifWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        def set_classif(device):
            controller.state.classif = device

        label = tk.Label(self, text="Select the classifier you want", background="#ADD8E6", font=small_times_font)
        label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        select_clasif_device1 = tk.Button(self, text="Damnini Doorbell",
                                          command=lambda:[controller.show_frame(SelectDataWindow), set_classif("Damnini_Doorbell")], height=3, width=20)
        select_clasif_device1.place(relx=0.3, rely=0.3, anchor=tk.CENTER)

        select_clasif_device2 = tk.Button(self, text="Ecobee Thermostat",
                                          command=lambda:[controller.show_frame(SelectDataWindow), set_classif("Ecobee_Thermostat")], height=3, width=20)
        select_clasif_device2.place(relx=0.3, rely=0.4, anchor=tk.CENTER)

        select_clasif_device3 = tk.Button(self, text="Ennio Doorbell",
                                          command=lambda:[controller.show_frame(SelectDataWindow), set_classif("Ennio_Doorbell")], height=3, width=20)
        select_clasif_device3.place(relx=0.3, rely=0.5, anchor=tk.CENTER)

        select_clasif_device4 = tk.Button(self, text="Philips Baby Monitor",
                                          command=lambda:[controller.show_frame(SelectDataWindow), set_classif("Philips_B120N10_Baby_Monitor")], height=3, width=20)
        select_clasif_device4.place(relx=0.3, rely=0.6, anchor=tk.CENTER)

        select_clasif_device5 = tk.Button(self, wraplength=100, text="Provision Security Camera - 737E",
                                          command=lambda:[controller.show_frame(SelectDataWindow), set_classif("Provision_PT_737E_Security_Camera")], height=3, width=20)
        select_clasif_device5.place(relx=0.3, rely=0.7, anchor=tk.CENTER)

        select_clasif_device6 = tk.Button(self, wraplength=100, text="Provision Security Camera - 838",
                                          command=lambda:[controller.show_frame(SelectDataWindow), set_classif("Provision_PT_838_Security_Camera")], height=3, width=20)
        select_clasif_device6.place(relx=0.7, rely=0.3, anchor=tk.CENTER)

        select_clasif_device7 = tk.Button(self, text="Samsung SNH",
                                          command=lambda:[controller.show_frame(SelectDataWindow), set_classif("Samsung_SNH")], height=3, width=20)
        select_clasif_device7.place(relx=0.7, rely=0.4, anchor=tk.CENTER)

        select_clasif_device8 = tk.Button(self, wraplength=100, text="SimpleHome Security Camera - 1002",
                                          command=lambda:[controller.show_frame(SelectDataWindow), set_classif("SimpleHome_1002_SecurityCamera")], height=3, width=20)
        select_clasif_device8.place(relx=0.7, rely=0.5, anchor=tk.CENTER)

        select_clasif_device9 = tk.Button(self, wraplength=100, text="SimpleHome Security Camera - 1003",
                                          command=lambda:[controller.show_frame(SelectDataWindow), set_classif("SimpleHome_1003_SecurityCamera")], height=3, width=20)
        select_clasif_device9.place(relx=0.7, rely=0.6, anchor=tk.CENTER)

        select_clasif_all = tk.Button(self, text="Trained on All",
                                      command=lambda:[controller.show_frame(SelectDataWindow), set_classif("all")], height=3, width=20)
        select_clasif_all.place(relx=0.7, rely=0.7, anchor=tk.CENTER)

        select_clasif_back = tk.Button(self, text="Back", height=2, width=10,
                                       command=lambda: controller.show_frame(HomeWindow),
                                       bg="#FF0000")
        select_clasif_back.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


# Tests the already trained KNN, DT and RF classifiers from a device on the testing data from any device
# uses wf.select_data
# "device" - the device from which the data will be loaded
# "classif" - the device from which the classifiers will be loaded
# in this window you select the "device" (the data you want to test on)
class SelectDataWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Select the data you want to run on", background="#ADD8E6", font=small_times_font)
        label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        select_data_device1 = tk.Button(self, text="Damnini Doorbell",
                                        command=lambda: self.open_window(controller, wf.select_data, "Damnini_Doorbell"), height=3, width=20)
        select_data_device1.place(relx=0.3, rely=0.3, anchor=tk.CENTER)

        select_data_device2 = tk.Button(self, text="Ecobee Thermostat",
                                        command=lambda: self.open_window(controller, wf.select_data, "Ecobee_Thermostat"), height=3, width=20)
        select_data_device2.place(relx=0.3, rely=0.4, anchor=tk.CENTER)

        select_data_device3 = tk.Button(self, text="Ennio Doorbell",
                                        command=lambda:self.open_window(controller, wf.select_data, "Ennio_Doorbell"), height=3, width=20)
        select_data_device3.place(relx=0.3, rely=0.5, anchor=tk.CENTER)

        select_data_device4 = tk.Button(self, text="Philips Baby Monitor",
                                        command=lambda: self.open_window(controller, wf.select_data, "Philips_B120N10_Baby_Monitor"), height=3, width=20)
        select_data_device4.place(relx=0.3, rely=0.6, anchor=tk.CENTER)

        select_data_device5 = tk.Button(self, wraplength=100, text="Provision Security Camera - 737E",
                                        command=lambda: self.open_window(controller, wf.select_data, "Provision_PT_737E_Security_Camera"), height=3, width=20)
        select_data_device5.place(relx=0.3, rely=0.7, anchor=tk.CENTER)

        select_data_device6 = tk.Button(self, wraplength=100, text="Provision Security Camera - 838",
                                        command=lambda: self.open_window(controller, wf.select_data, "Provision_PT_838_Security_Camera"), height=3, width=20)
        select_data_device6.place(relx=0.7, rely=0.3, anchor=tk.CENTER)

        select_data_device7 = tk.Button(self, text="Samsung_SNH",
                                        command=lambda: self.open_window(controller, wf.select_data, "Samsung_SNH"), height=3, width=20)
        select_data_device7.place(relx=0.7, rely=0.4, anchor=tk.CENTER)

        select_data_device8 = tk.Button(self, wraplength=100, text="SimpleHome Security Camera - 1002",
                                        command=lambda: self.open_window(controller, wf.select_data, "SimpleHome_1002_SecurityCamera"), height=3, width=20)
        select_data_device8.place(relx=0.7, rely=0.5, anchor=tk.CENTER)

        select_data_device9 = tk.Button(self, wraplength=100, text="SimpleHome Security Camera - 1003",
                                        command=lambda: self.open_window(controller, wf.select_data, "SimpleHome_1003_SecurityCamera"), height=3, width=20)
        select_data_device9.place(relx=0.7, rely=0.6, anchor=tk.CENTER)

        select_data_all = tk.Button(self, wraplength=100, text="All the data",
                                        command=lambda: self.open_window(controller, wf.select_data, "all"), height=3, width=20)
        select_data_all.place(relx=0.7, rely=0.7, anchor=tk.CENTER)

        select_data_back = tk.Button(self, text="Back", height=2, width=10,
                                     command=lambda: controller.show_frame(SelectClasifWindow),
                                     bg="#FF0000")
        select_data_back.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

    def open_window(self, controller, function, device):
        window = Window(self, controller, function, device)
        window.grab_set()


class NNWindow1(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        def set_classif(device):
            controller.state.classif = device

        label = tk.Label(self, text="Select the device you want", background="#ADD8E6", font=small_times_font)
        label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        nn1_device1 = tk.Button(self, text="Damnini Doorbell",
                                          command=lambda:[controller.show_frame(NNWindow2), set_classif("Damnini_Doorbell")], height=3, width=20)
        nn1_device1.place(relx=0.3, rely=0.3, anchor=tk.CENTER)

        nn1_device2 = tk.Button(self, text="Ecobee Thermostat",
                                          command=lambda:[controller.show_frame(NNWindow2), set_classif("Ecobee_Thermostat")], height=3, width=20)
        nn1_device2.place(relx=0.3, rely=0.4, anchor=tk.CENTER)

        nn1_device3 = tk.Button(self, text="Ennio Doorbell",
                                          command=lambda:[controller.show_frame(NNWindow2), set_classif("Ennio_Doorbell")], height=3, width=20)
        nn1_device3.place(relx=0.3, rely=0.5, anchor=tk.CENTER)

        nn1_device4 = tk.Button(self, text="Philips Baby Monitor",
                                          command=lambda:[controller.show_frame(NNWindow2), set_classif("Philips_B120N10_Baby_Monitor")], height=3, width=20)
        nn1_device4.place(relx=0.3, rely=0.6, anchor=tk.CENTER)

        nn1_device5 = tk.Button(self, wraplength=100, text="Provision Security Camera - 737E",
                                          command=lambda:[controller.show_frame(NNWindow2), set_classif("Provision_PT_737E_Security_Camera")], height=3, width=20)
        nn1_device5.place(relx=0.3, rely=0.7, anchor=tk.CENTER)

        nn1_device6 = tk.Button(self, wraplength=100, text="Provision Security Camera - 838",
                                          command=lambda:[controller.show_frame(NNWindow2), set_classif("Provision_PT_838_Security_Camera")], height=3, width=20)
        nn1_device6.place(relx=0.7, rely=0.3, anchor=tk.CENTER)

        nn1_device7 = tk.Button(self, text="Samsung SNH",
                                          command=lambda:[controller.show_frame(NNWindow2), set_classif("Samsung_SNH")], height=3, width=20)
        nn1_device7.place(relx=0.7, rely=0.4, anchor=tk.CENTER)

        nn1_device8 = tk.Button(self, wraplength=100, text="SimpleHome Security Camera - 1002",
                                          command=lambda:[controller.show_frame(NNWindow2), set_classif("SimpleHome_1002_SecurityCamera")], height=3, width=20)
        nn1_device8.place(relx=0.7, rely=0.5, anchor=tk.CENTER)

        nn1_device9 = tk.Button(self, wraplength=100, text="SimpleHome Security Camera - 1003",
                                          command=lambda:[controller.show_frame(NNWindow2), set_classif("SimpleHome_1003_SecurityCamera")], height=3, width=20)
        nn1_device9.place(relx=0.7, rely=0.6, anchor=tk.CENTER)

        nn1_all = tk.Button(self, text="Run on All the Data",
                                      command=lambda:[controller.show_frame(NNWindow2), set_classif("all")], height=3, width=20)
        nn1_all.place(relx=0.7, rely=0.7, anchor=tk.CENTER)

        nn1_back = tk.Button(self, text="Back", height=2, width=10,
                                       command=lambda: controller.show_frame(HomeWindow),
                                       bg="#FF0000")
        nn1_back.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


class NNWindow2(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Choose the Neural Network you want to use", background="#ADD8E6", font=small_times_font)
        label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # RELU SIMPLE
        relu_simple_label = tk.Label(self, text="RELU Activation \n SCALED data")
        relu_simple_label.place(relx=0.275, rely=0.2, anchor=tk.CENTER)
        relu_simple_25 = tk.Button(self, text="Epochs: 25", height=3, width=10,
                                       command=lambda: nn.create_model_relu(controller.state.classif, 25))
        relu_simple_25.place(relx=0.2, rely=0.3, anchor=tk.CENTER)

        relu_simple_50 = tk.Button(self, text="Epochs: 50", height=3, width=10,
                                       command=lambda: nn.create_model_relu(controller.state.classif, 50))
        relu_simple_50.place(relx=0.35, rely=0.3, anchor=tk.CENTER)

        relu_simple_75 = tk.Button(self, text="Epochs: 75", height=3, width=10,
                                       command=lambda: nn.create_model_relu(controller.state.classif, 75))
        relu_simple_75.place(relx=0.2, rely=0.4, anchor=tk.CENTER)

        relu_simple_100 = tk.Button(self, text="Epochs: 100", height=3, width=10,
                                       command=lambda: nn.create_model_relu(controller.state.classif, 100))
        relu_simple_100.place(relx=0.35, rely=0.4, anchor=tk.CENTER)


        # RELU NORMALIZED
        relu_norm_label = tk.Label(self, text="RELU Activation \n NORMALIZED data")
        relu_norm_label.place(relx=0.725, rely=0.2, anchor=tk.CENTER)
        relu_norm_25 = tk.Button(self, text="Epochs: 25", height=3, width=10,
                                       command=lambda: nn.create_model_relu_normalized(controller.state.classif, 25))
        relu_norm_25.place(relx=0.65, rely=0.3, anchor=tk.CENTER)

        relu_norm_50 = tk.Button(self, text="Epochs: 50", height=3, width=10,
                                       command=lambda: nn.create_model_relu_normalized(controller.state.classif, 50))
        relu_norm_50.place(relx=0.8, rely=0.3, anchor=tk.CENTER)

        relu_norm_75 = tk.Button(self, text="Epochs: 75", height=3, width=10,
                                       command=lambda: nn.create_model_relu_normalized(controller.state.classif, 75))
        relu_norm_75.place(relx=0.65, rely=0.4, anchor=tk.CENTER)

        relu_norm_100 = tk.Button(self, text="Epochs: 100", height=3, width=10,
                                       command=lambda: nn.create_model_relu_normalized(controller.state.classif, 100))
        relu_norm_100.place(relx=0.8, rely=0.4, anchor=tk.CENTER)


        # TANH SIMPLE
        tahn_simple_label = tk.Label(self, text="TAHN Activation \n SCALED data")
        tahn_simple_label.place(relx=0.275, rely=0.55, anchor=tk.CENTER)
        tahn_simple_25 = tk.Button(self, text="Epochs: 25", height=3, width=10,
                                       command=lambda: nn.create_model_tanh(controller.state.classif, 25))
        tahn_simple_25.place(relx=0.2, rely=0.65, anchor=tk.CENTER)

        tahn_simple_50 = tk.Button(self, text="Epochs: 50", height=3, width=10,
                                       command=lambda: nn.create_model_tanh(controller.state.classif, 50))
        tahn_simple_50.place(relx=0.35, rely=0.65, anchor=tk.CENTER)

        tahn_simple_75 = tk.Button(self, text="Epochs: 75", height=3, width=10,
                                       command=lambda: nn.create_model_tanh(controller.state.classif, 75))
        tahn_simple_75.place(relx=0.2, rely=0.75, anchor=tk.CENTER)

        tahn_simple_100 = tk.Button(self, text="Epochs: 100", height=3, width=10,
                                       command=lambda: nn.create_model_tanh(controller.state.classif, 100))
        tahn_simple_100.place(relx=0.35, rely=0.75, anchor=tk.CENTER)


        # TANH NORMALIZED
        tahn_norm_label = tk.Label(self, text="TANH Activation \n NORMALIZED data")
        tahn_norm_label.place(relx=0.725, rely=0.55, anchor=tk.CENTER)
        tahn_norm_25 = tk.Button(self, text="Epochs: 25", height=3, width=10,
                                       command=lambda: nn.create_model_tanh_normalized(controller.state.classif, 25))
        tahn_norm_25.place(relx=0.65, rely=0.65, anchor=tk.CENTER)

        tahn_norm_50 = tk.Button(self, text="Epochs: 50", height=3, width=10,
                                       command=lambda: nn.create_model_tanh_normalized(controller.state.classif, 50))
        tahn_norm_50.place(relx=0.8, rely=0.65, anchor=tk.CENTER)

        tahn_norm_75 = tk.Button(self, text="Epochs: 75", height=3, width=10,
                                       command=lambda: nn.create_model_tanh_normalized(controller.state.classif, 75))
        tahn_norm_75.place(relx=0.65, rely=0.75, anchor=tk.CENTER)

        tahn_norm_100 = tk.Button(self, text="Epochs: 100", height=3, width=10,
                                       command=lambda: nn.create_model_tanh_normalized(controller.state.classif, 100))
        tahn_norm_100.place(relx=0.8, rely=0.75, anchor=tk.CENTER)

        nn2_back = tk.Button(self, text="Back", height=2, width=10,
                                       command=lambda: controller.show_frame(NNWindow1),
                                       bg="#FF0000")
        nn2_back.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

        label2 = tk.Label(self, text="*Note that the results and additional information will be printed into the terminal*", background="#ADD8E6", wraplength=500)
        label2.place(relx=0.5, rely=0.95, anchor=tk.CENTER)


# class StatWindow(tk.Frame):
#     def __init__(self, parent, controller):
#         tk.Frame.__init__(self, parent)
#         label = tk.Label(self, text="Statistics", background="#ADD8E6", font=small_times_font)
#         label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
#
#         stat_back = tk.Button(self, text="Back", height=2, width=10,
#                               command=lambda: controller.show_frame(HomeWindow),
#                               bg="#FF0000")
#         stat_back.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


app = Detection()
app.iconphoto(True, tk.PhotoImage(file='icon.png'))
app.title("IOT Attacks Detection")
app.iconbitmap("icon1.ico")
app.mainloop()
