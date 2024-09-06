from torch.utils.data import TensorDataset, DataLoader

import MACROS
import parser
import torch


class Tester:
    def __init__(self, model, is_mdn=False):
        self.p = None
        self.model = model
        self.is_mdn = is_mdn

    def load_model(self, path):
        """
        Load model specified by path var
        :param path: path to saved model
        :return: nothing
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()  # Set model to evaluation mode if needed
        print(f"Model {path} loaded successfully.")

    def load_data(self, path):
        """
        Load testing data from path var
        :param path: path to training data csv
        :return: nothing
        """
        self.p = parser.Parser(MACROS.pts, path, MACROS.pos_dict)

    def test_player(self, player):
        if self.is_mdn:
            return self.mdn_test(player)
        return self.test(player)

    def mdn_test(self, player):
        """
        Test player stat for mdn network
        :param player: name of player to test
        :return: prediction value
        """
        if self.p is None:
            print("Data not loaded!\n ---> Call Tester.load_data to run!")
            return

        (x_train, y_train, x_test, y_test) = self.p.parse_one(player)

        # Creating the dataloaders for training
        test_dataset = TensorDataset(x_test.dataframe, y_test.dataframe)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        # Pass input through the model
        pi_pred = []
        sigma_pred = []
        mu_pred = []
        targets_list = []
        with torch.no_grad():
            self.model.eval()  # Set model to evaluation mode
            for batch_X, batch_y in test_loader:
                # Pass batch through the model
                val = batch_X.float()
                (pi, sigma, mu) = self.model(val)
                pi_pred.append(pi)
                sigma_pred.append(sigma)
                mu_pred.append(mu)
                targets_list.append(batch_y)
        return pi_pred, sigma_pred, mu_pred, targets_list

    def test(self, player):
        """
        Test player stat for normal network
        :param player: name of player to test
        :return: prediction value
        """
        if self.p is None:
            print("Data not loaded!\n ---> Call Tester.load_data to run!")
            return

        (x_train, y_train, x_test, y_test) = self.p.parse_one(player)

        # Creating the dataloaders for training
        test_dataset = TensorDataset(x_test.dataframe, y_test.dataframe)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        # Pass input through the model
        outputs = []
        targets_list = []
        with torch.no_grad():
            self.model.eval()  # Set model to evaluation mode
            for batch_X, batch_y in test_loader:
                # Pass batch through the model
                val = batch_X.float()
                outputs.append(self.model(val))
                targets_list.append(batch_y)
        return outputs, targets_list
