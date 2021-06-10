import threading
import time
import socket
from mlModel import MCLR

import json
import pickle
from struct import pack, unpack
import copy
import time


HOST = "127.0.0.1"
HANDSHAKE_SIZE = 1048
BUFF_SIZE = 4096
INIT_TIME = 30
T = 100


DEBUG = False
if DEBUG:
    INIT_TIME = 1
    T = 100
def debug(string):
    if DEBUG:
        print(f"debug: {string}")


class ClientHandler:
    def __init__(self, connection_tupple):
        (connection, (host, port)) = connection_tupple
        self.host = host
        self.port = port
        self.connection = connection
        self.current_iteration = 0;
        self.acc = None
        self.loss = None
        self.update_thread = None

        handshake = self.connection.recv(HANDSHAKE_SIZE)
        (self.id, self.data_size) = handshake.decode('utf-8').split(",")
        try:
            self.data_size = int(self.data_size)
        except:
            self.data_size = 0

        self.model = MCLR()

    def send_data(self, data, socket):
        msg = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

        length = pack('>Q', len(msg))

        self.connection.sendall(length)
        self.connection.sendall(msg)

    def recv_data(self, socket):
        # read in data
        bs = self.connection.recv(8)
        (length,) = unpack('>Q', bs)

        read_amount = 0
        remaining = length;
        msg = b"";
        while read_amount < length:
            msg += self.connection.recv(remaining)
            read_amount += len(msg)
            remaining = length - read_amount;

#        msg = self.connection.recv(length)
        data = pickle.loads(msg)
        return data

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

    # update_on_thread, gets an update on a seperate thread
    #
    # @param old_model, data to be sent to the client
    #
    # @return update_thread, returns the update thread
    #         after thread has started
    def update_on_thread(self, old_model):
        try:

            # SET PARAMETERS TO SERVER MODEL BEFORE SENDING
            self.set_parameters(old_model)

            update_thread = threading.Thread(target=self.get_update)
            update_thread.start()
            debug(f"\t{self.id}: update thread started")
            return update_thread
        except:
            debug(f"\t{self.id}: thread unable to start")
            return None

    # get_update, sends the current value of update to the client then waits
    # for a responce with the new update
    #
    # update set to None if update fails
    def get_update(self):
        try:
            debug(f"\t{self.id}: sending update")
            debug("Is model none? " + str(self.model == None))

            # send global server model data
            dict = {"model": self.model, "acc": self.acc, "loss": self.loss}
            self.send_data(dict, self.connection)

            print(f"Getting local model from client {self.id[-1]}")

            # recv client updated model data
            data = self.recv_data(self.connection)

            new_model = data["model"]
            new_acc = data["acc"]
            new_loss = data["loss"]

            # copy parameters of model
            self.set_parameters(new_model)

            self.acc = new_acc
            self.loss = new_loss

            debug(f"{self.acc}, {self.loss}")

            self.current_iteration += 1

            return

        except Exception as e:
            debug(f"\t{self.id}: error updating: " + str(e))
            self.model = None
            return


class Server:
    def __init__(self, id, port, max_clients):
        self.id = id
        self.port = port
        self.max_clients = max_clients


        self.clients = {}
        self.threads = None
        self.update_thread = None
        self.running = False
        self.model = None
        self.start_time = None
        self.end_time = None


    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            #print("setting", str(old_param), str(new_param))
            old_param.data = new_param.data.clone()

    # get_clients, returns a list of all current client handlers
    def get_clients(self):
        clients = []
        for port in self.clients:
            clients.append(self.clients[port])
        return clients

    # add_client, adds a client to the dictionary of current client handlers
    #
    # @param client, the client handler to add
    def add_client(self, client):
        self.clients[client.id] = client
        self.total_train_samples += client.data_size
        print(f"Client added: {client.id}")
        if len(self.clients) == 1:
            self.start_update_thread()

    # remove_client, removes a client from the client handler dictionary
    #
    # @param client, the client handler to remove
    def remove_client(self, client):
        if client.id in self.clients:
            debug(f"Client removed {client.id}")
            self.clients.pop(client.id)


    # starts the update process thread
    def start_update_thread(self):
        try:
            self.update_thread = threading.Thread(target=self.update_thread_method)
            self.update_thread.start()
            return
        except Exception as e:
            print("Error launching update thread: " + str(e))

    # wait INIT_TIME seconds then update continually
    def update_thread_method(self):
        print(f"First Client added. Updating in {INIT_TIME}s")

        timer = 0
        while timer < INIT_TIME and self.running:
            time.sleep(1)
            timer += 1
            if len(self.clients) == 5:
                print("5 Clients connected, Starting...")
                break

        self.start_time = time.time()

        for i in range(T):

            print(f"Global iteration {i + 1}:")

            #retrieve updated client local models from global model
            update_data = self.get_updates()

            print("Aggregating new global model")
            #updates the server model (aggregation and performance measure)
            self.on_updates(update_data)

            print("Broadcasting new global model")

            if not self.running:
                debug("update thread ending")
                return

        self.running = False
        self.end_time = time.time()
        debug("Done")
        print()
        print("Press \"CTR+C\" when you are ready to view the graphs and results!")
        return

    # gets updates from all clients in paralel
    #
    # @return update_data, a list of all updates from all clients
    def get_updates(self):
        clients = self.get_clients()

        debug(f"updating {len(clients)} clients")
        print(f"Total number of clients: {len(clients)}")

        #Get updates from all current clients in parrel
        self.threads = []
        for client in clients:
            self.threads.append(client.update_on_thread(self.model))

        #wait for all threads to finish
        for thread in self.threads:
            thread.join()

        #get the updates from all clients
        update_data = []
        for client in clients:
            #if update failed, remove client from client list
            if client.model == None:
                self.remove_client(client)
            else:
                update_data.append({
                    "model": client.model,
                    "id": client.id,
                    "data_size": client.data_size,
                    "acc": client.acc,
                    "loss": client.loss,
                    "current_iteration": client.current_iteration
                })

        debug(f"updated {len(update_data)} clients")

        return update_data

    # on_update, called once all updates are retreived
    #
    # @param data, a list of all updates from clients
    # @return new_model, the new aggregated model
    def on_update(self, data):
        pass

    def clean_threads(self):
        if isinstance(self.update_thread, threading.Thread):
            debug("joining main update thread")
            self.update_thread.do_run = False

        if isinstance(self.threads, list):
            debug("joining client update thread")
            for thread in self.threads:
                thread.do_run = False


    # start_socket, starts socket and waits for connections
    def start_socket(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

                # now listen for connections
                s.bind((HOST, self.port)) #listen on host address and port number
                s.listen(self.max_clients)

                self.running = True
                print(f"Server started and listening on {HOST}:{self.port}")

                while self.running:
                    # When a client connects, create a client handler
                    # and add it to clients
                    client = ClientHandler(s.accept())
                    self.add_client(client)
                s.close()

        except KeyboardInterrupt:
            debug("\nClosing socket")
            self.running = False
            self.clean_threads()
            s.close()
        except Exception as e:
            debug("Socket failed: " + str(e))
            self.running = False
            self.clean_threads()
            s.close()
