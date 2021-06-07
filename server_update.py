#SERVER UPDATE
from threading import Thread
import random
import sys
import time

IP = "127.0.0.1" # defaulted to localhost
start_time = time.time()

######################
# CLIENT FEDAVG CODE #
######################

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
from torch.utils.data import DataLoader, RandomSampler
import matplotlib
import matplotlib.pyplot as plt

######################

class MCLR(nn.Module):
    def __init__(self):
        super(MCLR, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_num_train_samples(id=""):
    train_path = os.path.join("FLdata", "train", "mnist_train_client" + str(id) + ".json")
    train_data = {}
    with open(os.path.join(train_path), "r") as f_train:
        train = json.load(f_train)
        train_data.update(train['user_data'])
    y_train = train_data['0']['y']
    y_train = torch.Tensor(y_train).type(torch.int64)
    train_samples = len(y_train)
    return train_samples


def get_data(id=""):
    train_path = os.path.join("FLdata", "train", "mnist_train_client" + str(id) + ".json")
    test_path = os.path.join("FLdata", "test", "mnist_test_client" + str(id) + ".json")
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


class UserAVG():
    def __init__(self, client_id, model, learning_rate, batch_size):
        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = get_data(client_id)
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]
        #self.sampler = RandomSampler(self.train_data, replacement=True, num_samples=1)
        #self.trainloader = DataLoader(self.train_data, batch_size, sampler=self.sampler)
        self.trainloader = DataLoader(self.train_data, batch_size, shuffle=True)
        self.testloader = DataLoader(self.test_data, self.test_samples)
        self.loss = nn.NLLLoss()
        self.model = copy.deepcopy(model)
        self.id = client_id
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)


    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()


#            num_batches =  self.train_samples // batch_size;
#            print(num_batches)
#
#            random.seed(a=None); #Seeded on time
            #select some random batch
#            selection = random.randint(0, num_batches - 1)
#            count = 0
            for batch_idx, (X, y) in enumerate(self.trainloader):
#                if count == 0:
#                    print(str(batch_idx), len(X))

                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
                break
#                count += 1
            #print(str(count), str(batch_idx))
            #print(len(X))
        return loss.data

    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0]).item()
        return test_acc

    def send_local_model(self):
        return

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


def client_update(client_id, client_losses, client_list, t):

    #THIS SHOULD BE RUN ONCE PER CLIENT START
    '''
    #overwrite performance log file for client
    #RUN ONCE PER SESSION
    file_name = "client" + str(client_id) + "_log.txt"
    cf = open(file_name, "w")
    cf.close()
    '''

    #update/train model and store loss into allocated client losses slot
    client_index = client_id - 1;
    client = client_list[client_index];

    print("I am client", str(client_id));
    print("Receiving new global model");

    #recieve global model
    #
    # Code to receive global model via socket
    # (instead of current method via. direct shared-memory transfer)
    #
    ####################


    #evaluate global model training loss
    for batch_idx, (X, y) in enumerate(client.testloader):
        output = client.model(X)
        loss = client.loss(output, y)
        break
    print("Training loss:", str(loss.data.item()));

    #test global model accuracy
    test_acc = client.test();
    print("Testing accuracy:", str(test_acc) + "%");

    #write global model performance to text file
    client.update_log_file(str(t), str(test_acc), str(loss.data.item()));

    print("Local training...");
    #train for 2 epochs and obtain loss as return value
    loss = client_list[client_index].train(2);

    print("Sending new local model");
    #send loss over to server
    #send new local model
    #
    # Code to send new local model via socket
    # (instead of current method via. direct transfer)
    #
    ####################
    client_losses[client_index] = loss; #this sends through shared memory

    return



##SERVER FUNCTIONS

def send_parameters(server_model, users):
    for user in users:
        user.set_parameters(server_model)

def aggregate_parameters(server_model, users, total_train_samples, M):
    #clear gobal model before aggregation
    for param in server_model.parameters():
        param.data = torch.zeros_like(param.data)

    if M == 0:
        for user in users:
            for server_param, user_param in zip(server_model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * user.train_samples / total_train_samples
        return server_model
    elif M > 0:

        random.seed(a=None); #Seeded on time
        possible_user_indexes = [0,1,2,3,4];

        for m in range(M):
            #random user index is chosen without replacement
            index = possible_user_indexes.pop(random.randint(0, len(possible_user_indexes) - 1))
            user = users[index]

            for server_param, user_param in zip(server_model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * user.train_samples / total_train_samples
        return server_model

def evaluate(users):
    total_accurancy = 0
    for user in users:
        total_accurancy += user.test()
    return total_accurancy/len(users)

def receive_local_model():
    return



def server_update(M, K, T, client_list, server_model, total_train_samples):


    # running FedAvg for T iterations and K clients
    loss = []
    acc = []
    for t in range(T):



        # fake broadcast for sending global model to all clients
        send_parameters(server_model, client_list)




        #evaluate the gobal model across all clients
        avg_acc = evaluate(client_list)
        acc.append(avg_acc)
        print(f"Global Round: {str(t + 1)} Average accuracy across all clients : {str(avg_acc)}")

        # each client keeps training process to  obtain new local model from the global model
        avgLoss = 0;
        client_threads = [];
        client_losses = [None] * K;
        client_id = 1 #CLIENT INDEX => client_id - 1
        for client in client_list:



            #replace with sending instruction via socket
            ct = Thread(target=client_update, args=(client_id, client_losses, client_list, t,));
            ct.start();
            client_threads.append(ct);
            client_id += 1;



        # wait for all threads to finish training local models
        for ct in client_threads:
            ct.join();

        #add the losses then divide them to get the avg
        for k in range(K):
            avgLoss += client_losses[k];
        avgLoss = avgLoss / K;


        loss.append(avgLoss)
        # aggregate all clients model to obtain new global model
        aggregate_parameters(server_model, client_list, total_train_samples, M)

    end_time = time.time()
    print(f"Final AVG Loss: {str(avgLoss.item())}\nFinal AVG Accuracy {str(avg_acc)}")
    print("Run-time was :", end_time - start_time)

    #Do final plots (LOSS)
    plt.figure(1,figsize=(5, 5))
    plt.plot(loss, label="FedAvg", linewidth  = 1)
    plt.legend(loc='upper right', prop={'size': 12}, ncol=2)
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.show()

    #Do final plots (ACCURACY)
    plt.figure(2,figsize=(5, 5))
    plt.plot(acc, label="FedAvg", linewidth  = 1)
    plt.ylim([0,  0.99])
    plt.legend(loc='upper right', prop={'size': 12}, ncol=2)
    plt.ylabel('Testing Acc')
    plt.xlabel('Global rounds')
    plt.show()




# Main server thread runs this
class Server():
    def __init__(self, port_num, M, K, T, client_list, server_model, total_train_samples):
        self.port_num = port_num
        self.M = M
        self.K = K
        self.T = T
        self.client_list = client_list
        self.server_model = server_model
        self.total_train_samples = total_train_samples

    def run(self):

        server_update(M, K, T, client_list, server_model, total_train_samples);


        #in actual, server binds to socket and communication to clients is via socket


        '''
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                print("Server host name:", IP, "Port: ", self.port_num)
                # Bind to the port
                s.bind((IP, self.port_num))
                # Start listening for connections with backlog of 10
                s.listen(10)

                while True:
                    #Check if this server has been deleted
                    if self.table[port_id][2] == -2:
                        break

                    #blocking call to accept a connection
                    conn, addr = s.accept()

                    #launch threads to handle new client connections
                    _thread.start_new_thread(on_new_client_node, (self.port_id, conn, addr, self.table))

                #exits here on node deletion
                s.close()
                print("You killed me!")
                raise KeyboardInterrupt()

        except KeyboardInterrupt:
            print()
            print("Keyboard Interrupt on Server... Exiting.")
            #set scheduler & updater threads to stop running

            scheduler_thread.do_run = False
            cf_updater_thread.do_run = False


        except:
            print("Server can't connect to the Socket")
            #set scheduler & updater threads to stop running

            scheduler_thread.do_run = False
            cf_updater_thread.do_run = False

        '''



if __name__ == "__main__":

    #   1. Verify and initialise command arguments
    if len(sys.argv) < 3:
        print("Too few command arguments provided")
        exit()
    if len(sys.argv) > 3:
        print("Too many command arguments provided")
        exit()

    # init port_num
    try:
        port_num = int(sys.argv[1])
        if port_num < 6000:
            print("Port number is < 6000, invalid!")
            exit()
    except:
        print("Invalid port number type given")
        exit()

    # init M => sub-sampling number
    try:
        sub_sampling = int(sys.argv[2])
        if sub_sampling < 0:
            print("Sub-sampling number is < 0, invalid!")
            exit()
    except:
        print("Invalid sub-sampling number type given")
        exit()

    #SERVER PARAMETERS
    if sub_sampling == 0: M = 0;
    elif sub_sampling == 1: M = 2;
    else: M = sub_sampling;

    T = 10;
    K = 5;
    num_user = 5;
    client_list = [];
    server_model = MCLR();
    total_train_samples = 0;



    #CLIENT PARAMETERS
    batch_size = 20;
    if M != 0: learning_rate = 0.0005
    if M == 0: learning_rate = 0.01 + 0.04# + 0.01 #- 0.005# - 0.004;
    OPTION = "MINIBATCHGD"; # "MINIBATCHGD" OR IF "GD"

    # create a federate learning network
    for k in range(K):


        #get client training data information, including number of samples
        if OPTION == "GD":
            #set batch_size to be full sample size
            batch_size = get_num_train_samples(k+1);

        #print("Total:", str(get_num_train_samples(k+1)), str(k+1))

        new_client = UserAVG(k+1, server_model, learning_rate, batch_size);
        total_train_samples += new_client.train_samples;
        client_list.append(new_client);


        #fake broadcast function
        send_parameters(server_model, client_list);


    #overwrite performance log file for clients
    #THIS SHOULD BE HANDLED BY THE CLIENT ONCE SERVER/CLIENT's ARE SPLIT
    for client_id in range(1, 6):
        file_name = "client" + str(client_id) + "_log.txt"
        cf = open(file_name, "w")
        cf.close()


#   4. Launch the server
    server = Server(port_num, M, K, T, client_list, server_model, total_train_samples);
    server.run();

    print("batch_size:", str(batch_size), "learning_rate:", str(learning_rate), "M:", str(M));
