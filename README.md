# COMP3221 Assignment 2

Instructions on how to run the code for this assignment are below.

## Program Structure

The following program structure must be maintained in the same directory to correctly run the programs:

* ***. /***
  * README.md
  * COMP3221_FLServer.py
  * COMP3221_FLClient.py
  * client.py
  * server.py
  * fedAvgClient.py
  * fedAvgServer.py
  * mlModel.py
  * FLData /
     * test /
          * mnist_test_client1.json
          * mnist_test_client2.json
          * mnist_test_client3.json
          * mnist_test_client4.json
          * mnist_test_client5.json
     * train /
          * mnist_train_client1.json
          * mnist_train_client2.json
          * mnist_train_client3.json
          * mnist_train_client4.json
          * mnist_train_client5.json


## Usage

### To run the Server

```bash
python3 COMP3221_FLServer.py 6000 <sub-sampling>
```


Allowable inputs

* <sub-sampling> = {0, 1}

example:

```bash
python3 COMP3221_FLServer.py 6000 0
```
---

### To run the 5 Clients

```bash
python3 COMP3221_FLClient.py <client_id> <port_num> <opt-method>
```

Allowable ***exclusive*** inputs

* <client_id> = {client1, client2, client3, client4, client5}
* <port_num> = {6001, 6002, 6003, 6004, 6005}
* <opt-method> = {0, 1}


example:

```bash
python3 COMP3221_FLClient.py client1 6001 0
```

## Output

The ***server*** waits for ***5 clients*** to connect or for ***30 seconds*** after the first client connects to begin iterating.

The ***client program***'s when run will create one "<client_id>_log.txt" text file in the same directory. The ***client program*** also prints the required messages to the command window.

The ***server program*** on exit (***using CTRL+C***) will output two graphs in sequence (opens after previous closes). One graph for loss over 100 training iterations and a second graph for accuracy over 100 training iterations. The ***server program*** also prints the required messages to the command window.

## Additional Notes

* Make sure the ***DEBUG*** flag is set to ***False*** (DEBUG = False) to avoid seeing excess prints

* The server program requires the user to press ***CTRL + C*** to ***exit and view the graphs***

* You must ***close the graphs*** for the ***server program*** to fully exit
