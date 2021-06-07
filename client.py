import threading
import time
import socket

DEBUG = True
def debug(string):
    if DEBUG:
        print(string)

HOST = "127.0.0.1"

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
                    update = s.recv(2048)
                    update = update.decode('utf-8')
                    debug("update received")

                    new_update = self.on_update(update)
                    time.sleep(3)
                    s.sendall(bytes(new_update, encoding='utf-8'))
                    debug(f"update sent {new_update}")

        except KeyboardInterrupt:
            debug("Client closing")
            return
        except:
            debug("Client unable to connect to server")
            return
