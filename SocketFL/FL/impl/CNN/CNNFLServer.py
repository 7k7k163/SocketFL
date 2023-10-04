import copy
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from utils import utils
from FL.impl.CNN.CNN import *
from FL.FLServer import FLServer


class CNNFLServer(FLServer):
    def __init__(self, conf, test_dataset):
        self.conf = conf
        self.w_locals = []
        self.global_model = self.new_model()
        self.eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf['batch_size_test'],
                                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                           np.random.choice(range(len(test_dataset)), len(test_dataset))))
        self.count = 0

    def local_update(self, model):
        self.w_locals.append(model)

    def model_aggregate(self):
        w_glob = utils.FedAvg(self.w_locals)
        self.global_model.load_state_dict(w_glob)
        # self.model_eval()
        # for i in range(4):
        #     self.global_model = utils.load_model(self.global_model, self.w_locals[i])
        #     self.model_eval()

    def model_eval(self):
        self.count += 1
        model = self.get_model()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]
            if self.conf['cuda']:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        print('turn [' + str(self.count) + ']: acc: ' + str(acc) + ', loss: ' + str(total_l))

    def new_model(self):
        if self.conf['type'] == 'mnist':
            model = CNN_MNIST()
        elif self.conf['type'] == 'cifar':
            model = CNN_CIFAR()
        if self.conf['cuda']:
            model.cuda()
        return model

    def get_model(self):
        model = copy.deepcopy(self.global_model)
        if self.conf['cuda']:
            model.cuda()
        return model
