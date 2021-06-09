from client import Client
from mlModel import MLModel, MCLR

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import json


class FedAvgClient(Client):
    def __init__(self, id, opt_method, learning_rate, batch_size):
        super(FedAvgClient, self).__init__(id)
        self.opt_method = opt_method;
        self.data_size = self.get_num_train_samples();
        self.local_iteration = 0;
        self.learning_rate = learning_rate;
        self.client_model = MCLR();
        self.id = id;

        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = self.get_data()
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]
        self.trainloader = DataLoader(self.train_data, batch_size, shuffle=True)
        self.testloader = DataLoader(self.test_data, self.test_samples)
        self.optimizer = torch.optim.SGD(self.client_model.parameters(), lr=learning_rate)
        self.loss = nn.NLLLoss();


        if opt_method == 0: #if GD
            self.batch_size = self.data_size;
        elif opt_method == 0: #if MINIBATCHGD
            self.batch_size = self.batch_size;



    def run(self):

        self.create_log_file();

        self.connect_socket();



    def on_update(self, server_model):
        # create new model from global server model

        self.client_model = server_model;

        self.local_iteration += 1;

        print("I am client", str(self.id[-1]));
        print("Receiving new global model");

        # evaluate global server model training loss
        for batch_idx, (X, y) in enumerate(self.testloader):
            output = self.model(X)
            loss = self.loss(output, y)
            break
        print("Training loss:", str(loss.data.item()));

        # test global server model accuracy
        acc = self.test();
        print("Testing accuracy:", str(acc) + "%");

        #write global model performance to text file
        self.update_log_file(str(self.local_iteration), str(acc), str(loss.data.item()));


        print("Local training...");
        #train for 2 epochs and obtain loss as return value
        loss = self.train(2);

        print("Sending new local model");

        return self.client_model;


    def train(self, epochs):
        for epoch in range(epochs):
            self.client_model.train()

            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
                break
        return loss.data


    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0]).item()
        return test_acc


    def update_log_file(self, t, acc, loss):
        #append updated performance log file for client
        file_name = "client" + str(self.id) + "_log.txt"
        cf = open(file_name, "a")

        log_file_lines = ["Client: " + str(self.id) + ", Round: " + t,
        "Training loss: " + loss,
        "Testing accuracy: " + acc + "%"];

        for line in log_file_lines:
            cf.write(line + "\n")
        cf.write("\n")
        cf.close()








    def get_num_train_samples(self):
        train_path = os.path.join("FLdata", "train", "mnist_train_" + str(self.id) + ".json")
        train_data = {}
        with open(os.path.join(train_path), "r") as f_train:
            train = json.load(f_train)
            train_data.update(train['user_data'])
        y_train = train_data['0']['y']
        y_train = torch.Tensor(y_train).type(torch.int64)
        train_samples = len(y_train)
        return train_samples

    def get_data(self):
        train_path = os.path.join("FLdata", "train", "mnist_train_" + str(self.id) + ".json")
        test_path = os.path.join("FLdata", "test", "mnist_test_" + str(self.id) + ".json")
        train_data = {}
        test_data = {}

        with open(os.path.join(train_path), "r") as f_train:
            train = json.load(f_train)
            train_data.update(train['user_data'])
        with open(os.path.join(test_path), "r") as f_test:
            test = json.load(f_test)
            test_data.update(test['user_data'])

        X_train, y_train, X_test, y_test = train_data['0']['x'], train_data['0']['y'], test_data['0']['x'], test_data['0']['y']
        X_train = torch.Tensor(X_train).view(-1, 1, 28, 28).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, 1, 28, 28).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
        train_samples, test_samples = len(y_train), len(y_test)
        return X_train, y_train, X_test, y_test, train_samples, test_samples

    def create_log_file(self):
        file_name = str(self.id) + "_log.txt"
        cf = open(file_name, "w")
        cf.close()
