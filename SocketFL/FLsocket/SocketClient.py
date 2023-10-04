import socket
from threading import Thread, RLock
from utils import utils


class SocketClient(object):
    def __init__(self, ip, port, buffer=200000):
        self.lock = RLock()
        self.ip = ip
        self.connected = False
        self.port = port
        self.buffer = buffer
        self.client_socket = None
        self.thread = None

    def connect(self):
        self.lock.acquire()
        try:
            if not self.connected:
                self.client_socket = socket.create_connection((self.ip, self.port))
                # data_dict = dict()
                # data_dict['status'] = 1
                # data_dict['data'] = {'Bandwidth': np.random.randint(1, 10), 'Mode': ['4G', 'WIFI'][np.random.randint(0, 2)]}
                data_dict = utils.make_dict(1, {''})
                self.client_socket.send(bytes('{}'.format(data_dict), 'utf-8'))
                self.thread = ThreadConnect(self)
                self.thread.start()
                self.connected = True
        finally:
            self.lock.release()

    def disconnect(self):
        self.lock.acquire()
        try:
            if self.connected:
                self.client_socket.close()
                # self.thread.close()
                utils.stop_thread(self.thread)
                self.thread = None
                self.connected = False
        finally:
            self.lock.release()

    def handle_data(self, data):
        if self.connected:
            print('server: ' + data)

    def a__handle_data(self, data):
        self.lock.acquire()
        self.handle_data(data)
        self.lock.release()

    def a__initialize(self, data):
        self.lock.acquire()
        self.initialize(data)
        self.lock.release()

    def initialize(self, data):
        pass


class ThreadConnect(Thread):
    def __init__(self, client):
        super().__init__()
        self.client = client

    def run(self):
        while True:
            try:
                data = self.client.client_socket.recv(self.client.buffer)
                data = eval(data.decode())
                if data['status'] == 2:
                    # print('初始化！')
                    self.client.a__initialize(data)
                elif data['status'] == 3:
                    # print('训练过程！')
                    self.client.a__handle_data(data['data'])
                elif data['status'] == 4:
                    self.client.client_socket.close()
                    self.client.disconnect()
            # except ConnectionError:
            except:
                # self.client.disconnect()
                break
