import threading
import time
import socket
import pickle

HOST = "127.0.0.1"
HANDSHAKE_SIZE = 1048
INIT_TIME = 1

DEBUG = True
def debug(string):
    if DEBUG:
        print(f"debug: {string}")


class ClientHandler:
    def __init__(self, connection_tuple):
        (connection, (host, port)) = connection_tuple;
        self.connection = connection;
        self.host = host;
        self.port = port;

        handshake = self.connection.recv(HANDSHAKE_SIZE)
        (self.id, self.data_size) = handshake.decode('utf-8').split("|")
        debug(f"Received handshake ({self.id}, {self.data_size})")

        try:
            self.data_size = int(self.data_size)
        except:
            debug("Error converting data size to integer")
            self.data_size = 0

        self.server_model = None;
        self.client_model = None;


    # update_on_thread, gets an update on a seperate thread
    #
    # @param old_model, data to be sent to the client
    #
    # @return update_thread, returns the update thread
    #         after thread has started
    def update_on_thread(self, old_model):
        try:
            self.server_model = old_model
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
    #    try:
        debug(f"\t{self.id}: sending update")

        msg = pickle.dumps(self.server_model)

        self.connection.sendall(msg)
        print(f"Getting local model from client {self.id[-1]}")

        new_update = self.connection.recv(2048)

        debug(f"\t{self.id}: received update")
        new_update = pickle.loads(new_update)
        self.client_model = new_update
        #except:
        #    debug(f"\t{self.id}: error updating")
        #    self.client_model = None

class Server:
    def __init__(self, id, port, max_clients, T):
        self.id = id;
        self.port = port;
        self.max_clients = max_clients;
        self.T = T;


        self.clients = {};
        self.threads = None;
        self.update_thread = None;
        self.running = False;
        self.server_model = None;


    # get_clients, returns a list of all current client handlers
    def get_clients(self):
        clients = []
        for port in self.clients:
            clients.append(self.clients[port])
        return clients

    # add_client, adds a client to the dictionary of current client handlers
    # {key: value} => {"client1": client_handler_object}
    #
    # @param client, the client handler to add
    def add_client(self, client):
        self.clients[client.id] = client
        debug(f"Client added {client.id}")
        if len(self.clients) == 1:
            self.start_update_thread()

    # remove_client, removes a client from the client handler dictionary
    #
    # @param client, the client handler to remove
    def remove_client(self, client):
        if client.id in self.clients:
            debug(f"Client removed {client.id}")
            self.clients.pop(client.id)


    # starts the model update thread (model updating thread)
    def start_update_thread(self):
        try:
            self.update_thread = threading.Thread(target=self.update_thread_method)
            self.update_thread.start()
        except:
            debug("Failed to start update thread")

    # wait INIT_TIME seconds then update for T iterations
    def update_thread_method(self):
        debug(f"update thread started, updating in {INIT_TIME}s")

        timer = 0;
        while timer < INIT_TIME and self.running:
            time.sleep(1)
            timer += 1

        for t in range(self.T):
            if t != 0:
                print("Broadcasting new global message")

            print(f"Global iteration {t + 1}:")

            # retrieve the client models
            updates = self.get_updates()

            # aggregate the client models
            self.server_model = self.on_updates(self.clients)

            if not self.running:
                debug("update thread ending")
                return
        self.running = False
        debug("Done")

    # gets updates from all clients in parallel
    #
    # @return update_data, a list of all updates from all clients
    def get_updates(self):
        clients = self.get_clients()

        debug(f"updating {len(clients)} clients")
        print(f"Total number of clients: {len(clients)}")

        #Get updates from all current clients in parallel
        threads = []
        for client in clients:
            threads.append(client.update_on_thread(self.server_model))

        #wait for all threads to finish
        for thread in threads:
            thread.join()

        '''
        #get the updates from all clients
        update_data = []
        for client in clients:
            #if update failed, remove client from client list
            if client.client_model == None:
                self.remove_client(client)
                debug("An update failed and a client was removed")
            else:
                update_data.append(client.client_model)

        debug(f"updated {len(update_data)} clients")

        return update_data
        '''

    # on_update, called once all updates are retreived
    #
    # @param data, a list of all updates from clients
    # @return new_model, the new aggregated model
    def on_updates(self, data):
        debug("Aggregating new global model")
        debug(data)
        return "new_model"

    def clean_threads(self):
        if isinstance(self.update_thread, threading.Thread):
            debug("joining main update thread")
            self.update_thread.join()

        if isinstance(self.threads, list):
            debug("joining client update thread")
            for thread in self.threads:
                thread.join()


    # start_socket, starts socket and waits for connections
    def start_socket(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

                # now listen for connections
                s.bind((HOST, self.port)) #listen on host address and port number
                s.listen(self.max_clients)

                self.running = True
                debug(f"Socket started and listening on {HOST}:{self.port}")

                while self.running:
                    # When a client connects, create a client handler
                    # and add it to clients
                    client = ClientHandler(s.accept())
                    self.add_client(client)

        except KeyboardInterrupt:
            debug("\nClosing socket")
            self.running = False
            self.clean_threads()
        except:
            debug("Socket failed")
            self.running = False
            self.clean_threads()
