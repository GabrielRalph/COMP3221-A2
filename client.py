import threading
import time
import socket

DEBUG = True
def debug(string):
    if DEBUG:
        print(string)

HOST = "127.0.0.1"
MODEL_SIZE = 4096

class Client:
    def __init__(self, id, server_port):
        self.id = id
        self.data_size = 100
        self.port = server_port

    @property
    def handshake(self):
        return f"{self.id},{self.data_size}"

    def on_update(self, old_update):
        new_update = "some new_update"
        #do something

        return new_update

    def connect_socket(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

                s.connect((HOST, self.port))

                #send id
                s.sendall(bytes(self.handshake, encoding='utf-8'))

                while 1:
                    update = s.recv(MODEL_SIZE)
                    update = update.decode('utf-8')
                    debug("update received")

                    new_update = self.on_update(update)
                    s.sendall(new_update)
                    debug(f"update sent")

        except KeyboardInterrupt:
            debug("Client closing")
            return
        except:
            debug("Client unable to connect to server")
            return
