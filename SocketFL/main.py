import json
import time

from FLsocket.SocketFLClient import SocketFLClient
from FLsocket.SocketFLServer import SocketFLServer
from FL.impl.CNN.CNNFLServer import CNNFLServer
from FL.impl.MLP.MLPFLServer import MLPFLServer
from utils import datasets

if __name__ == '__main__':
    server = None
    with open("./config.json", 'r') as f:
        conf = json.load(f)
    train_datasets, test_datasets = datasets.get_dataset("./data/", conf["type"])
    if conf['model'] == 'cnn':
        server = SocketFLServer(12306, 2000000, CNNFLServer(conf, test_datasets))
        for i in range(conf['k']):
            client = SocketFLClient('127.0.0.1', 12306, 2000000)
    elif conf['model'] == 'mlp':
        server = SocketFLServer(12306, 2000000, MLPFLServer(conf, test_datasets))
        for i in range(conf['k']):
            client = SocketFLClient('127.0.0.1', 12306, 2000000)
    while not server.closed:
        time.sleep(5)
