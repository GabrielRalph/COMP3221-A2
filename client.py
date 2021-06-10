import threading
import time
import socket

import json
import pickle
from struct import pack, unpack

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
        pass

    def connect_socket(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

                s.connect((HOST, self.port))

                #send id
                s.sendall(bytes(self.handshake, encoding='utf-8'))

                while 1:

                    #raw_recv = data_recv.decode('utf-8')

                    #msg_data = bytes(msg, encoding= 'utf-8')

                    bs = s.recv(8)
                    (length,) = unpack('>Q', bs)
                    debug(length)

                    msg = s.recv(length)
                    data = pickle.loads(msg)
                    debug(data)


                    #recv'd server model, update it
                    update = data
                    new_update = self.on_update(update)

                    #send back to server
                    msg = pickle.dumps(new_update)

                    length = pack('>Q', len(msg))
                    s.sendall(length)
                    s.sendall(msg)
                    debug("done sending updated local model")

                    #msg_size = s.recv(4)
                    #msg_size = msg_size.decode('utf-8')
                    #print(msg_size)

                    #data = []
                    #while True:
                    #    packet = s.recv(MODEL_SIZE)
                    #    if not packet: break
                    #    data.append(packet)

                    #print("here")
                    #data_arr = pickle.loads(b"".join(data))

                    #data = raw_recv.decode('utf-8')
                    #print("here")
                    #update = pickle.loads(raw_recv)
                    #debug(update)

                    #update = s.recv(MODEL_SIZE)
                    #print("here")
                    #update = json.loads(update)

                    #debug("update received")
                    #debug(data_arr)


                    #new_update = self.on_update(update)
                    #s.sendall(new_update)
                    #debug(f"update sent")

        except KeyboardInterrupt:
            debug("Client closing")
            s.close()
            return
        except Exception as e:
            debug("Client unable to connect to server: " + str(e))
            s.close()
            return
