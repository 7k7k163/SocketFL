from FL.impl.CNN.CNNFLClient import CNNFLClient
from FL.impl.MLP.MLPFLClient import MLPFLClient
from FLsocket.SocketClient import SocketClient
from utils import utils, datasets


class SocketFLClient(SocketClient):
    def __init__(self, ip, port, buffer):
        self.client = None
        self.conf = None
        super().__init__(ip, port, buffer)
        self.connect()

    def initialize(self, data):
        self.conf = data['conf']
        # CNNFLClient(conf, train_datasets, test_datasets, i)
        train_datasets, test_datasets = datasets.get_dataset("./data/", self.conf["type"])
        if self.conf['model'] == 'cnn':
            self.client = CNNFLClient(self.conf, train_datasets, test_datasets, data['serial_number'])
        elif self.conf['model'] == 'mlp':
            self.client = MLPFLClient(self.conf, train_datasets, test_datasets, data['serial_number'])

    def handle_data(self, data):
        model = utils.load_model(self.client.new_model(), data)
        diff = self.client.local_train(model)
        # buffer = dict()
        # buffer['status'] = 3
        # buffer['data'] = utils.save_model(diff, './model_file/' + str(self.client.client_id) + '.pt')
        data_dict = utils.make_dict(3, [utils.save_model(diff, './model_file/' + str(self.client.client_id) + '.pt')])
        self.client_socket.send(bytes('{}'.format(data_dict), 'utf-8'))
