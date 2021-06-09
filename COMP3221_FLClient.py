from fedAvgClient import FedAvgClient
from mlModel import MLModel

import sys

if __name__ == "__main__":

    #   1. Verify and initialise command arguments
    if len(sys.argv) < 4:
        print("Too few command arguments provided")
        exit()
    if len(sys.argv) > 4:
        print("Too many command arguments provided")
        exit()

    # init client id
    client_id = sys.argv[1]

    # init port_num
    try:
        port_num = int(sys.argv[2])
        if port_num < 6000:
            print("Port number is < 6000, invalid!")
            exit()
    except:
        print("Invalid port number type given")
        exit()

    # init opt_method => opt-method number
    try:
        opt_method = int(sys.argv[3])
        if opt_method != 0 and opt_method != 1:
            print("Opt-method number can only be 0 or 1, invalid!")
            exit()
    except:
        print("Invalid opt-method number type given")
        exit()



    #    2. Initialise Client Parameters
    learning_rate = 0.01;
    batch_size = 20;


    #   3. Launch Client
    client = FedAvgClient(client_id, opt_method, learning_rate, batch_size);
    client.run();
