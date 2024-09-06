import torch
import torch.nn as nn
import torch.nn.functional as f


# Transformer Model 1
class Transformer1(nn.Module):
    def __init__(self, n_head, num_encoder_layers, num_decoder_layers,
                 input_size, hidden_size, output_size, num_lstm_layers, dropout=0.5):
        """
        Uses transformer encoder to "highlight" specific parts of input sequence
        Passes sequence through LSTM layer in parallel
        Combines and normalizes outputs to then be directed to linear layers to form output
        :param n_head: number of encoder attention heads
        :param num_encoder_layers: number of encoder layers
        :param num_decoder_layers: number of decoder layers
        :param input_size: number of input features
        :param hidden_size: number of hidden lstm layers
        :param output_size: number of outputs
        :param num_lstm_layers: number of lstm layers
        :param dropout: dropout probability, default 0.5
        """
        super(Transformer1, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(input_size, n_head, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(input_size, n_head, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)

        self.lstm = nn.LSTM(input_size, hidden_size, num_lstm_layers, batch_first=True)

        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.lin1 = nn.Linear(input_size, output_size)
        self.lin2 = nn.Linear(hidden_size, output_size)
        self.output_layer = nn.Linear(output_size * 2, output_size)

    def forward(self, inputs):
        """
        Forward propagation through transformer then lstm
        :param inputs: input batch
        :return: output prediction
        """
        # start with transformer encoding
        out1 = self.transformer_encoder(inputs)

        # decode transformer
        out1 = self.transformer_decoder(out1, inputs)

        # send normal input to LSTM
        out2, _ = self.lstm(inputs)

        # feed output of both LSTM and transformer to linear output layers
        # Transpose the output to match the shape expected by BatchNorm1d
        out1 = out1.transpose(1, 2)
        out2 = out2.transpose(1, 2)

        # Apply batch normalization
        out1 = self.batch_norm1(out1)
        out2 = self.batch_norm2(out2)

        # Transpose back to the original shape
        out1 = out1.transpose(1, 2)
        out2 = out2.transpose(1, 2)

        out1 = f.relu(out1)
        out2 = f.relu(out2)

        out1 = self.lin1(out1[:, -1, :])
        out2 = self.lin2(out2[:, -1, :])

        out = torch.cat((out1, out2), 0)
        ret = []
        # get length of out
        size = out.size(0)
        for i in range(0, int(size / 2)):
            val = torch.stack((out[i], out[i + int(size / 2)]), 1)
            ret.append(self.output_layer(val))

        ret = torch.stack(ret)
        return ret


# Transformer Model 1
class Transformer2(nn.Module):
    def __init__(self, n_head, num_encoder_layers, num_decoder_layers,
                 input_size, hidden_size, output_size, num_lstm_layers, dropout=0.5):
        """
        Uses transformer encoder to "highlight" specific parts of input sequence
        Then passes modified sequence through LSTM layer defined my the NeuralNet class
        :param n_head: number of encoder attention heads
        :param num_encoder_layers: number of encoder layers
        :param num_decoder_layers: number of decoder layers
        :param input_size: number of input features
        :param hidden_size: number of hidden lstm layers
        :param output_size: number of outputs
        :param num_lstm_layers: number of lstm layers
        :param dropout: dropout probability, default 0.5
        """
        super(Transformer2, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(input_size, n_head, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(input_size, n_head, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)

        self.lstm = NeuralNet(input_size, hidden_size, output_size, num_lstm_layers)

    def forward(self, inputs):
        """
        Forward propagation through transformer then lstm
        :param inputs: input batch
        :return: output prediction
        """
        # start with transformer encoding
        out = self.transformer_encoder(inputs)

        # decode transformer
        out = self.transformer_decoder(out, inputs)

        # send normal input to LSTM
        out = self.lstm(out)

        return out


# Convolutional LSTM cell architecture
class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_games, dropout_prob=0.5):
        """
        Definition of LSTM architecture where input sequence is modified by volatility score of player
            Using a convolutional layer
        Also contains two linear output layers to finalize output
        :param input_size: Number of features to input
        :param hidden_size: Number of hidden states in LSTM
        :param output_size: Number of outputs
        :param num_layers: Number of layers in LSTM
        :param num_games: Number of games in sequence
        :param dropout_prob: Parameter to help with over-fitting, default = 0.5
        """
        super(ConvLSTM, self).__init__()

        # init general params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_games = num_games
        self.dropout = nn.Dropout(p=dropout_prob)

        # initialize convolutional layer (convolve in 2-dimensions over sequence of data)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 1), stride=(2, 1))

        # initialize LSTM network
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # initialize output layers
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Batch normalization layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        """
        Forward propagate through convolutional, lstm, and linear layers
        Input needs to be formatted such that every column of input is followed by a column of volatility
            score for the player being tested
        Make sure add_volatility function passed to input before forward propagation!!!
        :param inputs: features of player games
        :return: predicted value
        """

        # need to convolve over every sequence in batch
        out = []
        for i in inputs:
            i = torch.reshape(i, (1, self.num_games * 2, self.input_size))
            out.append(torch.reshape(self.conv(i), (self.num_games, self.input_size)))
        out = torch.stack(out)

        # send to lstm layer
        # hidden_states = torch.zeros(self.num_layers, out.shape[0], self.hidden_size)
        # cell_states = torch.zeros(self.num_layers, out.shape[0], self.hidden_size)
        out, _ = self.lstm(out)

        # Apply dropout
        out = self.dropout(out)

        # Transpose the output to match the shape expected by BatchNorm1d
        out = out.transpose(1, 2)

        # Apply batch normalization
        out = self.batch_norm(out)

        # Transpose back to the original shape
        out = out.transpose(1, 2)

        # Apply activation function
        out = f.relu(out)

        # send to output layer
        out = self.output_layer(out[:, -1, :])

        # return
        return out
def add_volatility(inputs, num_games, input_size, volatility):
    """
    Add volatility rows to input data for convolution
    :param inputs: player game sequences
    :param num_games: number of games in sequence
    :param input_size: number of features
    :param volatility: volatility rating of player
    :return: modified inputs (copy)
    """
    # basically pair each player game statistic with the player's volatility score
    #       basically append a column to each column of player data?
    #       This is necessary for convolution since the kernel is going to convolve with each statistic
    #       and the volatility score with some trainable "frame" to emphasize patterns
    # format input so that for every timestep(row) there is a row of the volatility value to convolve with
    inputs = torch.clone(inputs)
    out = []
    for x in inputs:
        # for each row insert new row of
        for i in range(1, num_games * 2, 2):
            top = x[0:i, :]
            low = x[i:, :]
            new = torch.full(size=(1, input_size), fill_value=volatility)
            # combine
            x = torch.cat((top, new, low), dim=0)
        out.append(x)

    # covert out into tensor
    out = torch.stack(out)

    return out
