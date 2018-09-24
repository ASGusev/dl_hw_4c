import numpy as np
import torch
import tensorboardX


def _walk_through_data(net, data, training, loss_function, optimizer=None):
    losses = []
    for x, y in data:
        if training:
            optimizer.zero_grad()
        out = net(x)
        loss = loss_function(out, y)
        losses.append(loss.item())
        if training:
            loss.backward()
            optimizer.step()
    return np.mean(losses)


class Trainer:
    def __init__(self, data_train, data_test, logging_path, lr=1e-3):
        self.data_train = data_train
        self.data_test = data_test
        self.logger = tensorboardX.SummaryWriter(logging_path)
        self.lr = lr
        self.loss_function = torch.nn.CrossEntropyLoss()

    def train(self, net, n_epochs):
        optimizer = torch.optim.Adam(net.parameters(), self.lr)
        for epoch in range(n_epochs):
            loss_train = _walk_through_data(net, self.data_train, True, self.loss_function, optimizer)
            loss_test = _walk_through_data(net, self.data_test, False, self.loss_function)
            self.logger.add_scalar('train_loss', loss_train)
            self.logger.add_scalar('test_loss', loss_test)
