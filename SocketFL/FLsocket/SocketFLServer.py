from FLsocket.SocketServer import SocketServer
import random
from utils import utils


class SocketFLServer(SocketServer):
    def __init__(self, port, buffer, server):
        self.global_epochs = 0
        self.server = server
        self.now_clients = None
        self.choose_clients = None
        super().__init__(port, buffer)
        self.open()

    def handle_data(self, client_thread, data):
        if self.now_clients is None or client_thread not in self.now_clients:
            return

        self.now_clients.remove(client_thread)
        self.server.local_update(data)

        if len(self.now_clients) == 0:
            self.global_epochs += 1
            self.server.model_aggregate()
            self.server.model_eval()
            if self.server.conf['global_epochs'] == self.global_epochs:
                self.close()
                return
            if len(self.clients) >= self.server.conf['k']:
                self.send_model()

    def handle_disconnect(self, client_thread):
        super().handle_disconnect(client_thread)
        if self.now_clients and client_thread in self.now_clients:
            self.now_clients.remove(client_thread)

    def handle_connect(self, client_thread):
        super().handle_connect(client_thread)
        print(client_thread.params)
        if len(self.clients) == self.server.conf['k']:
            self.send_model()

    def send_model(self):
        while True:
            self.now_clients = random.sample(self.clients, self.server.conf['d'])
            self.choose_clients = list()
            data_dict = utils.make_dict(3, [utils.save_model(self.server.get_model(),
                                                             './model_file/' + self.server.conf['type'] + '.pt')])
            for client in self.now_clients:
                try:
                    client.client.send(bytes('{}'.format(data_dict), 'utf-8'))
                    self.choose_clients.append(client)
                finally:
                    ...
            if len(self.choose_clients) != len(self.now_clients):
                print('未完全发送成功')
                self.now_clients = list()
                for client in self.choose_clients:
                    self.now_clients.append(client)
            if len(self.choose_clients) != 0:
                break
