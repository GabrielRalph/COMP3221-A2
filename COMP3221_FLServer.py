from fedAvgServer import FedAvgServer

import matplotlib.pyplot as plt
import sys
import time


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

    K = 5;
    batch_size = 20;
    learning_rate = 0.01;

    #   3. Launch Server
    try:
        server = FedAvgServer("server", port_num, K, M);
        server.run();
    except KeyboardInterrupt:
        server.running = False;
        server.clean_threads();


    #   4. Print Results
    print()
    print()
    print()
    print(f"Final AVG Loss: {str(server.avg_loss)}\nFinal AVG Accuracy {str(server.avg_acc)}")
    if server.end_time and server.start_time:
        print("Run-time was :", server.end_time - server.start_time)
    print("Batch size:", str(batch_size), "Learning rate:", str(learning_rate), "Sub-sampling:", str(M));

    #Do final plots (LOSS)
    plt.figure(1,figsize=(5, 5))
    plt.plot(server.loss, label="FedAvg", linewidth  = 1)
    plt.legend(loc='upper right', prop={'size': 12}, ncol=2)
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.show()

    #Do final plots (ACCURACY)
    plt.figure(2,figsize=(5, 5))
    plt.plot(server.acc, label="FedAvg", linewidth  = 1)
    plt.ylim([0,  0.99])
    plt.legend(loc='upper right', prop={'size': 12}, ncol=2)
    plt.ylabel('Testing Acc')
    plt.xlabel('Global rounds')
    plt.show()
