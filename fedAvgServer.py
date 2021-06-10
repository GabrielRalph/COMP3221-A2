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

DEBUG = False
def debug(string):
    if DEBUG:
        print(f"debug: {string}")


class FedAvgServer(Server):
    def __init__(self, id, port, max_clients, M):
        super(FedAvgServer, self).__init__(id, port, max_clients);
        self.total_train_samples = 0;
        self.M = M;
        # initialise model
        self.model = MCLR();

        self.acc = []
        self.loss = []
        self.avg_loss = 0
        self.avg_acc = 0
        self.global_round = 0

    def run(self):

        #start socket, add clients and begin iterating
        self.start_socket();


    def on_updates(self, models):

        # models = [client_index] {
        #   "model": client.model,
        #   "id": client.id,
        #   "data_size": client.data_size,
        #   "acc": client.acc,
        #   "loss": client.loss,
        #   "current_iteration": client.current_iteration
        # } ]

        self.global_round += 1

        # calculate avg accuracy
        self.avg_acc = self.evaluate(models)
        self.acc.append(self.avg_acc)
        debug(f"Global Round: {str(self.global_round)} Average accuracy across all clients : {str(self.avg_acc)}")


        # calculate avg loss
        self.avg_loss = 0;
        for cd in models:
            self.avg_loss += cd["loss"];
        self.avg_loss = self.avg_loss / len(models);

        self.loss.append(self.avg_loss)

        debug("acc: " + str(self.avg_acc))
        debug("loss: " + str(self.avg_loss))


        # clear gobal model before aggregation
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        # aggregate client models and return new server model
        if self.M == 0:
            for cd in models:
                for server_param, user_param in zip(self.model.parameters(), cd["model"].parameters()):
                    server_param.data = server_param.data + user_param.data.clone() * cd["data_size"] / self.total_train_samples
            return

        elif self.M > 0:

            random.seed(a=None); #Seeded on time
            #possible client indexes (5 clients) = [0,1,2,3,4];
            client_indexes = [x for x in range(0, len(models))]

            # incase of less clients than sub-sampling number
            if len(models) >= self.M:
                n_selections = self.M;
            else:
                n_selections = len(models);

            for m in range(n_selections):
                #random client index is chosen without replacement
                if len(client_indexes) == 1:
                    cd = models[0]
                else:
                    index = client_indexes.pop(random.randint(0, len(client_indexes) - 1))
                    cd = models[index]

                for server_param, user_param in zip(self.model.parameters(), cd["model"].parameters()):
                    server_param.data = server_param.data + user_param.data.clone() * cd["data_size"] / self.total_train_samples

            return


    def evaluate(self, models):
        total_accurancy = 0
        for cd in models:
            total_accurancy += cd["acc"]
        return total_accurancy/len(models)
