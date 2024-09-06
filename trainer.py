import MACROS
import network
import torch
import math
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    def __init__(self, model: network.NeuralNet, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.losses = []

    def train(self, x_train: torch.tensor, y_train: torch.tensor):
        if isinstance(self.optimizer, torch.optim.LBFGS):
            self.train_lbfgs(x_train, y_train)
        else:
            self.train_grad(x_train, y_train)

    def train_grad(self, x_train: torch.tensor, y_train: torch.tensor):
        """
        Trains network on data from given tensors with grad descent optimization
        :param x_train: training data
        :param y_train: training labels
        :return: nothing
        """
        # Creating the dataloaders for training
        train_dataset = TensorDataset(x_train.dataframe, y_train.dataframe)
        train_loader = DataLoader(train_dataset, batch_size=MACROS.batch_size, shuffle=True)

        # Starting training loop
        for epoch in range(MACROS.num_epochs):
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()

                # Forward Pass
                val = batch_X.float()
                outputs = self.model(val)  # Forward pass

                # Calculate Loss
                if self.criterion is None:
                    (pi, sigma, mu) = outputs
                    loss = network.loss(mu, sigma, pi, batch_y.float().view(batch_y.shape[0], 1), dist=MACROS.dist)
                else:
                    loss = self.criterion(outputs, batch_y.float().view(batch_y.shape[0], 1))

                # check if valid loss
                if math.isnan(loss.item()):
                    # throw exception
                    raise ValueError("Nan value encountered ending training!")

                # Backward pass and optimization
                loss.backward()

                self.optimizer.step()

            # Print epoch loss
            self.losses.append(loss.item())
            print(f'Epoch [{epoch + 1}/{MACROS.num_epochs}], Loss: {loss.item()}')

    def train_lbfgs(self, x_train: torch.tensor, y_train: torch.tensor):
        """
        Trains network on data from given tensors with lbfgs optimization
        :param x_train: training data
        :param y_train: training labels
        :return: nothing
        """
        # Creating the dataloaders for training
        train_dataset = TensorDataset(x_train.dataframe, y_train.dataframe)
        train_loader = DataLoader(train_dataset, batch_size=MACROS.batch_size, shuffle=True)

        # Starting training loop
        for epoch in range(MACROS.num_epochs):
            min_loss = []
            for batch_X, batch_y in train_loader:
                min_loss = []

                def closure():
                    self.optimizer.zero_grad()

                    # Forward Pass
                    val = batch_X.float()
                    outputs = self.model(val)  # Forward pass

                    # Calculate Loss
                    if self.criterion is None:
                        (pi, sigma, mu) = outputs
                        loss = network.loss(mu, sigma, pi, batch_y.float().view(batch_y.shape[0], 1), dist=MACROS.dist)
                    else:
                        loss = self.criterion(outputs, batch_y.float().view(batch_y.shape[0], 1))

                    # check if valid loss
                    if math.isnan(loss.item()):
                        # throw exception
                        raise ValueError("Nan value encountered ending training!")

                    # Backward pass and optimization
                    loss.backward()
                    min_loss.append(loss.item())

                    return loss

                self.optimizer.step(closure)

            # Print epoch loss
            if len(min_loss) != 0:
                self.losses.append(min(min_loss))
                print(f'Epoch [{epoch + 1}/{MACROS.num_epochs}], Loss: {min(min_loss)}')
