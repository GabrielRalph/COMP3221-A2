from client import Client
from mlModel import MLModel

class FedAvgClient(Client):
    def __init__(self, id, server_port, params):
        super(FedAvgClient, self).__init__(id, server_port)

    def run(self):
        self.data_size = 1024
        self.connect_socket()

    def on_update(self, w_t):
        # convert pickle data w_t into a model

        # update model ** TODO **

        # convert updated model to pickle data
        # return pickle data
        return "pickle"



# client = FedAvgClient("me", 6000, 5)
# client.run()
