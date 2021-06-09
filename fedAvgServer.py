from server import Server
from mlModel import MLModel, MCLR


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
from torch.utils.data import DataLoader, RandomSampler
import matplotlib
import matplotlib.pyplot as plt
import pickle

#unpickle
def up(data):
    return pickle.loads(data)

#pickle
def p(data):
    return pickle.dumps(data)

class FedAvgServer(Server):
    def __init__(self, id, port, max_clients, M):
        super(FedAvgServer, self).__init__(id, port, max_clients);
        self.total_train_samples = 0;
        self.M = M;
        # initialise model
        self.model = MCLR();
        #PICKLE THE MODEL
        self.model = p(self.model);
        self.acc = []
        self.loss = []
        self.avg_loss = 0
        self.avg_acc = 0
        self.global_round = 0

    def run(self):

        #start socket and add clients
        self.start_socket();




    def on_updates(self, models):

        #UNPICKLE MODEL
        MODEL = up(self.model)
        self.global_round += 1

        # calculate avg accuracy
        self.avg_acc = self.evaluate(models)
        self.acc.append(self.avg_acc)
        print(f"Global Round: {str(self.global_round)} Average accuracy across all clients : {str(self.avg_acc)}")

        # calculate avg loss
        self.avgLoss = 0;
        for m in models:
            self.avgLoss += m["loss"];
        self.avgLoss = self.avgLoss / len(models);

        self.loss.append(self.avgLoss)


        # clear gobal model before aggregation
        for param in MODEL.parameters():
            param.data = torch.zeros_like(param.data)

        # aggregate client models and return new server model
        if self.M == 0:
            for cd in models:
                for server_param, user_param in zip(up(self.model).parameters(), up(cd["model"]).parameters()):
                    server_param.data = server_param.data + user_param.data.clone() * cd["data_size"] / self.total_train_samples
            return p(self.model)
        elif self.M > 0:

            random.seed(a=None); #Seeded on time
            #possible_user_indexes = [0,1,2,3,4];
            possible_user_indexes = [x for x in range(0, len(models))]


            for m in range(self.M):
                #random user index is chosen without replacement
                index = possible_user_indexes.pop(random.randint(0, len(possible_user_indexes) - 1))
                cd = models[index]

                for server_param, user_param in zip(up(self.model).parameters(), up(cd["model"]).parameters()):
                    server_param.data = server_param.data + user_param.data.clone() * cd["data_size"] / self.total_train_samples
            return p(self.model)


    def evaluate(self, models):
        total_accurancy = 0
        for clients in self.clients.values():
            total_accurancy += clients.test()
        return total_accurancy/len(self.clients)
