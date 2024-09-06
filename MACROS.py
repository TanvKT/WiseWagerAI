# define macros
# Parameters for the model
current_size = 8
input_size = 11
time_steps = 5
output_size = 1
hidden_size = 50
num_dist = 20
n_head = input_size
num_encoder_layers = 10
num_decoder_layers = 10
num_lstm_layers = 150
num_games = 5
batch_size = 5
num_epochs = 50
learning_rate = 0.005
pts = 10

# More parameters for regularization
l2_regularization_strength = 0.005
l1_regularization_strength = 0.0005

# distribution type, loss type, and optimization type
dist = 'laplace'
loss = 'none'
optim = 'nadam'

pos_dict = {'GK': 1, 'CB': 2, 'LB': 3, 'RB': 3, 'WB': 4, 'DM': 5, 'CM': 6, 'LM': 6, 'RM': 6, 'LW': 7, 'RW': 7, 'FW': 8, 'AM' : 9}
