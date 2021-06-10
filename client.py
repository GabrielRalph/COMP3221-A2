import threading
import time
import socket

import json
import pickle
from struct import pack, unpack

DEBUG = False
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
        self.local_iteration = 0;


    @property
    def handshake(self):
        return f"{self.id},{self.data_size}"

    def on_update(self, old_update):
        pass

    def unpickle(self, data, socket):
        unpickled_data = None
        try:
            unpickled_data = pickle.loads(data)
            return unpickled_data
        except Exception as e:
            debug(f"Failed to unpickle: " + str(e))
            data += socket.recv(1)
            self.unpickle(data, socket)

    def connect_socket(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

                s.connect((HOST, self.port))

                #send handshake
                s.sendall(bytes(self.handshake, encoding='utf-8'))

                while self.local_iteration < 100:


                    bs = s.recv(8)
                    if not bs: break;
                    (length,) = unpack('>Q', bs)
                    debug(f"Server packet size: {length}")

                    # read in data

                    read_amount = 0
                    remaining = length - read_amount;
                    msg = b"";
                    while read_amount < length:
                        msg += s.recv(remaining)
                        read_amount += len(msg)
                        remaining = length - read_amount;
                    debug(f"Read in data: {str(len(msg))} bytes")
                    # unpickle data
                    #data = self.unpickle(msg, s)
                    data = pickle.loads(msg)


                    debug("Unpickled msg")

                    #recv'd server model, update it
                    update = data
                    new_update = self.on_update(update)

                    #send back to server
                    msg = pickle.dumps(new_update, protocol=pickle.HIGHEST_PROTOCOL)

                    length = pack('>Q', len(msg))
                    s.sendall(length)
                    s.sendall(msg)
                    debug("done sending updated local model")



        except KeyboardInterrupt:
            debug("Client closing")
            s.close()
            return
        except Exception as e:
            debug("Client unable to connect to server: " + str(e))
            s.close()
            return
