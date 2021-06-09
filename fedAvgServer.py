from server import Server
from mlModel import MLModel

class FedAvgServer(Server):
    def __init__(self, id, port, max_clients, params):
        super(FedAvgServer, self).__init__(id, port, max_clients)


    def run(self):
        self.model = "initialise model"
        self.start_socket()

    def on_update(self, models):
        # aggregate

        return MLModel()


fedAvg = FedAvgServer("server", 6000, 6, 123)
fedAvg.run()
