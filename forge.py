import MACROS
import parser
import trainer
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn


class Forge:
    def __init__(self, path: str, model, op_fcn='adam', l_fcn='mse'):
        self.model = model

        # define optimizer based on args
        if op_fcn == 'adadelta':
            optimizer = optim.Adadelta(self.model.parameters(), lr=MACROS.learning_rate)
        elif op_fcn == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=MACROS.learning_rate)
        elif op_fcn == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=MACROS.learning_rate)
        elif op_fcn == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=MACROS.learning_rate)
        elif op_fcn == 'sparse_adam':
            optimizer = optim.SparseAdam(self.model.parameters(), lr=MACROS.learning_rate)
        elif op_fcn == 'adamax':
            optimizer = optim.Adamax(self.model.parameters(), lr=MACROS.learning_rate)
        elif op_fcn == 'asgd':
            optimizer = optim.ASGD(self.model.parameters(), lr=MACROS.learning_rate)
        elif op_fcn == 'lbfgs':
            optimizer = optim.LBFGS(self.model.parameters(), lr=MACROS.learning_rate)
        elif op_fcn == 'nadam':
            optimizer = optim.NAdam(self.model.parameters(), lr=MACROS.learning_rate)
        elif op_fcn == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=MACROS.learning_rate)
        elif op_fcn == 'rprop':
            optimizer = optim.Rprop(self.model.parameters(), lr=MACROS.learning_rate)
        elif op_fcn == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=MACROS.learning_rate)
        else:
            print("Optimizer not defined, Defaulting ADAM")
            optimizer = optim.Adam(self.model.parameters(), lr=MACROS.learning_rate)

        ######################################################################
        # Define other loss functions if needed
        ######################################################################
        # define criterion based on input
        if l_fcn == "mse":
            criterion = nn.MSELoss()
        elif l_fcn == "none":
            criterion = None
        else:
            print("Loss function not defined, Defaulting mse")
            criterion = nn.MSELoss()

        # init parser and trainer
        self.p = parser.Parser(MACROS.pts, path, MACROS.pos_dict)
        (self.x_trains, self.y_trains, x_tests, y_tests) = self.p.parse_all()
        self.t = trainer.Trainer(self.model, criterion, optimizer)

        self.index = 0

    def train_one(self, player: str):
        """
        Trains network on one player
        :param player: name of player
        :return: nothing
        """
        try:
            index = list(self.p.dic.keys()).index(player)
        except ValueError:
            print(f'Player: {player} not found!!! Skipping train...')
            return []

        print(f'Player: {player} ----- {index}')

        self.t.train(self.x_trains[index], self.y_trains[index])
        return self.t.losses

    def train_all(self):
        """
        Trains network on all players
        :return: nothing
        """
        for player in self.p.dic:
            self.train_one(player)
            self.index = self.index + 1
        return self.t.losses

    def train_some(self, n: int):
        """
        Trains n players
        :param n: num players to train
        :return: nothing
        """
        for player in self.p.dic:
            if self.index is n:
                break
            self.train_one(player)
            self.index = self.index + 1
        return self.t.losses


def plot(losses):
    # print graph of losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses, marker='o', linestyle='-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss over Epoch')
    plt.show()
