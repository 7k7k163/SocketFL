import socket
from threading import Thread, RLock
from utils import utils
import time


class SocketServer(object):
    def __init__(self, port, buffer=200000, backlog=5):
        self.lock = RLock()
        self.opened = False
        self.closed = False
        self.port = port
        self.buffer = buffer
        self.backlog = backlog
        self.server_socket = None
        self.thread = None
        self.clients = None
        self.cid = 1

    def open(self):
        self.lock.acquire()
        try:
            if not self.opened:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.bind(('127.0.0.1', self.port))
                self.server_socket.listen(self.backlog)
                self.thread = ThreadListen(self)
                self.clients = list()
                self.thread.start()
                self.opened = True
                print("服务器已打开！")
        finally:
            self.lock.release()

    def close(self):
        self.lock.acquire()
        try:
            if self.opened:
                self.server_socket.close()
                # self.thread.close()
                utils.stop_thread(self.thread)
                self.thread = None
                self.opened = False
                for i in self.clients:
                    i.client.send(bytes('{}'.format({'status': 4}), 'utf-8'))
                    i.client.close()
                    i.server.a__handle_disconnect(i)
                self.clients = None
                print("服务器已关闭！")
                self.closed = True
        finally:
            self.lock.release()

    def handle_data(self, client_thread, data):
        if client_thread in self.clients:
            print('client [' + str(client_thread.uid) + ']: ' + data)

    def a__handle_data(self, client_thread, data):
        self.lock.acquire()
        try:
            self.handle_data(client_thread, data)
        finally:
            self.lock.release()

    def handle_connect(self, client_thread):
        self.clients.append(client_thread)
        print('client [' + str(client_thread.uid) + '] 已连接')

    def a__handle_connect(self, client_thread):
        self.lock.acquire()
        try:
            self.handle_connect(client_thread)
        finally:
            self.lock.release()

    def handle_disconnect(self, client_thread):
        # self.clients.remove(client_thread)
        print('client [' + str(client_thread.uid) + '] [' + str(client_thread.address) + '] 已断开连接')

    def a__handle_disconnect(self, client_thread):
        self.lock.acquire()
        try:
            self.handle_disconnect(client_thread)
        finally:
            self.lock.release()


class ThreadListen(Thread):
    def __init__(self, server):
        super().__init__(daemon=True)
        self.server = server
        self.open = True
        self.lock = RLock()
        self.uid = 1

    def run(self):
        while self.open:
            client, address = self.server.server_socket.accept()
            data = client.recv(self.server.buffer)
            data = eval(data.decode())
            if data['status'] == 1:
                data_dict = utils.make_dict(2, [self.uid, self.server.server.conf])
                client.send(bytes('{}'.format(data_dict), 'utf-8'))
                thread = ThreadMessage(self.uid, client, self.server, address, data['data'])
                thread.start()
                self.uid += 1
                self.server.a__handle_connect(thread)


class ThreadMessage(Thread):
    def __init__(self, uid, client, server, address, params):
        super().__init__()
        self.client = client
        self.server = server
        self.address = address
        self.uid = uid
        self.params = params

    def run(self):
        while True:
            try:
                data = self.client.recv(self.server.buffer)
                data = eval(data.decode())
                if data['status'] == 3:
                    self.server.a__handle_data(self, data['data'])
                time.sleep(5)
            # except ConnectionError and OSError:
            except:
                # self.server.a__handle_disconnect(self)
                break
