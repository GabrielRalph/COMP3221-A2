import threading
import time
import socket
import pickle

DEBUG = True
def debug(string):
    if DEBUG:
        print(string)

HOST = "127.0.0.1"
SERVER_PORT = 6000

class Client:
    def __init__(self, id):
        self.id = id;
        self.port = SERVER_PORT;
        self.data_size = 0;

    @property
    def handshake(self):
        return f"{self.id}|{self.data_size}"

    def on_update(self, server_model):
        pass


    def connect_socket(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

                s.connect((HOST, self.port))

                # send handshake message
                s.sendall(bytes(self.handshake, encoding='utf-8'))

                while 1:

                    print("HERERERERE")


                    update = s.recv(2048)
                    print("HERERERERE")


                    update = pickle.loads(update)


                    print("HERERERERE")

                    #update = update.decode('utf-8')
                    debug("update received")
                    debug(update)

                    if update == None:
                        print("NONE")
                        exit()

                    new_update = self.on_update(update)

                    msg = pickle.dumps(new_update)
                    s.sendall(msg)

                    debug(f"update sent {new_update}")

        except KeyboardInterrupt:
            debug("Client closing")
            return
        except:
            debug("Client unable to connect to server")
            return
