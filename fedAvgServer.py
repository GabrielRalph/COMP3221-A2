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


class FedAvgServer(Server):
    def __init__(self, id, port, max_clients, T, M):
        super(FedAvgServer, self).__init__(id, port, max_clients, T);
        self.total_train_samples = 0;
        self.T = T;
        self.M = M;
        # initialise model
        self.server_model = MCLR();

    def run(self):

        #start socket and add clients
        self.start_socket();




    def on_updates(self, clients):
        # aggregate client models and return new server model

        #clear gobal model before aggregation
        for param in self.server_model.parameters():
            param.data = torch.zeros_like(param.data)

        if self.M == 0:
            for client in clients.values():
                for server_param, user_param in zip(self.server_model.parameters(), client.client_model.parameters()):
                    server_param.data = server_param.data + user_param.data.clone() * client.data_size / self.total_train_samples
            return server_model
        elif self.M > 0:

            random.seed(a=None); #Seeded on time
            possible_user_indexes = [0,1,2,3,4];

            for m in range(self.M):
                #random user index is chosen without replacement
                index = possible_user_indexes.pop(random.randint(0, len(possible_user_indexes) - 1))
                user = users[index]

                for server_param, user_param in zip(self.server_model.parameters(), client.client_model.parameters()):
                    server_param.data = server_param.data + user_param.data.clone() * client.data_size / self.total_train_samples
            return server_model


        return MLModel()
