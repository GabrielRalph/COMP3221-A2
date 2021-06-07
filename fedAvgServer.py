from server import Server
from mlModel import MLModel

class FedAvgServer(Server):

    def run(self):
        self.model = "initialise model"
        self.start_socket()

    def on_update(self, models):
        # aggregate

        return MLModel()


fedAvg = FedAvgServer("server", 6000, 6)
fedAvg.run()
