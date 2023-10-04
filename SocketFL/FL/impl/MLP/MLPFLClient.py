import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from FL.FLClient import FLClient
from FL.impl.MLP.MLP import MLP


class MLPFLClient(FLClient):
    def __init__(self, conf, train_dataset, test_dataset, cid):
        self.conf = conf
        self.client_id = cid
        self.train_dataset = train_dataset
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        indices = all_range[cid * data_len: (cid + 1) * data_len]
        random.shuffle(indices)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=conf["batch_size_train"],
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
        self.eval_loader = DataLoader(test_dataset, batch_size=conf['batch_size_test'],
                                      sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                          np.random.choice(range(len(test_dataset)), len(test_dataset))))
        self.count = 0

    def local_train(self, model):
        new_model = copy.deepcopy(model)
        if self.conf['cuda']:
            new_model.cuda()
        optimiser = torch.optim.Adam(new_model.parameters(), lr=self.conf['lr'])
        new_model.train()
        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                if self.conf['cuda']:
                    data = data.cuda()
                    target = target.cuda()
                optimiser.zero_grad()
                output = new_model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimiser.step()
            print(self.client_id, 'Epoch {:d} done.'.format(e))
        m1, n1 = self.local_eval(new_model)
        print(self.client_id, "acc:", m1, ", loss:", n1, " After")
        return new_model

    def local_eval(self, model):
        model.eval()
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
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        return acc, total_l

    def new_model(self):
        model = None
        if self.conf['type'] == 'mnist':
            model = MLP(28*28*3, 120, 10)
        elif self.conf['type'] == 'cifar':
            model = MLP(32 * 32 * 3, 120, 10)
        if self.conf['cuda']:
            model.cuda()
        return model
