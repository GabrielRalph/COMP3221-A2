from client import Client
from mlModel import MLModel

class FedAvgClient(Client):

    def run(self):
        self.data_size = 1024
        self.connect_socket()

    def on_update(self, w_t):
        # create new model

        return MLModel()


client = FedAvgClient("me", 6000)
client.run()
