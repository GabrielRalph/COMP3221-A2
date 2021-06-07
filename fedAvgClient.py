from client import Client

class fedAvgClient(Client):
    
    def run(self):
        self.data_size = 1024
        self.connect_socket()

    def on_update(self, w_t):
        # create new model

        return "w(k){t + 1}"
