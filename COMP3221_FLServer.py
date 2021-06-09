from fedAvgServer import FedAvgServer
from mlModel import MLModel

import sys

if __name__ == "__main__":

    #   1. Verify and initialise command arguments
    if len(sys.argv) < 3:
        print("Too few command arguments provided")
        exit()
    if len(sys.argv) > 3:
        print("Too many command arguments provided")
        exit()

    # init port_num
    try:
        port_num = int(sys.argv[1])
        if port_num < 6000:
            print("Port number is < 6000, invalid!")
            exit()
    except:
        print("Invalid port number type given")
        exit()

    # init M => sub-sampling number
    try:
        sub_sampling = int(sys.argv[2])
        if sub_sampling < 0:
            print("Sub-sampling number is < 0, invalid!")
            exit()
    except:
        print("Invalid sub-sampling number type given")
        exit()

    #    2. Initialise Server Parameters
    if sub_sampling == 0: M = 0;
    elif sub_sampling == 1: M = 2;
    else: M = sub_sampling;

    T = 100;
    K = 5;

    #   3. Launch Server
    server = FedAvgServer("server", port_num, K, T, M);
    server.run();
