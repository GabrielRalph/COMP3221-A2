from server import Server

class FedAvgServer(Server):

    def run(self):
        self.model = "initialise model"
        self.start_socket()

    def on_update(self, models):
        # aggregate

        return "w{t + 1}"


fedAvg = FedAvgServer("server", 6000, 6)
fedAvg.run()
