import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class PandasDataset(Dataset):
    def __init__(self, dataframe):
        # This converts the dataframe passed into a numpy array and then into a pytorch tensor
        self.dataframe = torch.tensor(dataframe.values)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return self.dataframe[index]


class Parser:
    def __init__(self, pts: int, path: str, pos_dict: dict):
        # Preprocessing the data to get a dictionary of players with their corresponding data needed for
        # training
        print("----Preprocessing Data----")
        df = pd.read_csv(path)
        df['Pos.y'] = df['Pos.y'].map(pos_dict)
        df = df.groupby("Player")
        player_dic = {}
        for group in df:
            player_dic[group[0]] = group[1]
        for player in player_dic:
            player_dic[player] = player_dic[player].drop(["Match_Date", "Player", "Home_Team", "Away_Team", "Succ_percent_Take_Ons"], axis=1)

        # Not considering players that have data points lesser than pts
        self.dic = {}
        for player in player_dic:
            if (player_dic[player].shape[0]) >= pts:
                self.dic[player] = player_dic[player]
        print("----Preprocessing Successful----")

    def parse_one(self, player) -> tuple:
        """
        Parses data from one player in csv to pytorch tensors
        :param player: player to parse
        :return: tuple in for X_train, y_train, X_test, y_test
        """
        df = self.dic[player]
        X = df.drop("Sh", axis=1)
        y = df[["Sh"]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors
        X_train_tensor = PandasDataset(X_train)
        y_train_tensor = PandasDataset(y_train)
        X_test_tensor = PandasDataset(X_test)
        y_test_tensor = PandasDataset(y_test)

        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

    def parse_all(self) -> tuple:
        """
            Parses all data in generated dic
            :return: all tensors in list form (X_train, Y_train, X_test, y_test)
        """
        print("----Parsing Data----")
        # This dictionary holds the trained model for each corresponding player
        x_trains = []
        y_trains = []
        x_tests = []
        y_tests = []
        for player in self.dic:
            (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor) = self.parse_one(player)

            # add to lists
            x_trains.append(X_train_tensor)
            y_trains.append(y_train_tensor)
            x_tests.append(X_test_tensor)
            y_tests.append(y_test_tensor)

        # return in tuple form
        print("----Parsing Complete----")
        return x_trains, y_trains, x_tests, y_tests
