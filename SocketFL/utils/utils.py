import inspect
import ctypes
import io
import numpy as np
import torch


def _async_raise(tid, ex_ctypes):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(ex_ctypes):
        ex_ctypes = type(ex_ctypes)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(ex_ctypes))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


def save_model(model, _dir):
    torch.save(model.state_dict(), _dir)
    fo = open(_dir, "rb")
    s = fo.read()
    fo.close()
    return s


def load_model(model, buffer):
    d = torch.load(io.BytesIO(buffer))
    for k in model.state_dict():
        model.state_dict()[k].copy_(d[k])
    return model


def FedAvg(w):
    w = [torch.load(io.BytesIO(i)) for i in w]
    w_avg = w[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def make_dict(status, d):
    data_dict = dict()
    if status == 1:
        data_dict['status'] = 1
        data_dict['data'] = {'Bandwidth': np.random.randint(1, 10), 'Mode': ['4G', 'WIFI'][np.random.randint(0, 2)]}
    elif status == 2:
        data_dict['status'] = 2
        data_dict['serial_number'] = d[0]
        data_dict['conf'] = d[1]
    elif status == 3:
        data_dict['status'] = 3
        data_dict['data'] = d[0]
    return data_dict
