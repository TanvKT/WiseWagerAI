import forge
import network
import MACROS


"""
    Example script to use PredictorModel files
    This script will make and train a model on the data for one player 
        in the Premier League from 2017 to 2024
"""

# start the forge
model = network.NeuralNet(MACROS.current_size, MACROS.input_size, MACROS.hidden_size, MACROS.output_size, MACROS.num_lstm_layers, MACROS.num_dist, dist=MACROS.dist)
print("----------2017-2018----------")
f = forge.Forge(r'../data/Train_Data/2017-2018_PL_train.csv', model, op_fcn=MACROS.optim, l_fcn=MACROS.loss)
losses = []

###################################################################################
# If you want to train on one player
# f.train_one(<player name>)
# You can also then train on another player by running the same function
###################################################################################

try:
    # train on specific player
    losses = losses + f.train_one("Aaron Cresswell")

    # repeat for each season
    print("----------2018-2019----------")
    f = forge.Forge(r'../data/Train_Data/2018-2019_PL_train.csv', model, op_fcn=MACROS.optim, l_fcn=MACROS.loss)
    losses = losses + f.train_one("Aaron Cresswell")

    print("----------2019-2020----------")
    f = forge.Forge(r'../data/Train_Data/2019-2020_PL_train.csv', model, op_fcn=MACROS.optim, l_fcn=MACROS.loss)
    losses = losses + f.train_one("Aaron Cresswell")

    print("----------2020-2021----------")
    f = forge.Forge(r'../data/Train_Data/2020-2021_PL_train.csv', model, op_fcn=MACROS.optim, l_fcn=MACROS.loss)
    losses = losses + f.train_one("Aaron Cresswell")

    print("----------2021-2022----------")
    f = forge.Forge(r'../data/Train_Data/2021-2022_PL_train.csv', model, op_fcn=MACROS.optim, l_fcn=MACROS.loss)
    losses = losses + f.train_one("Aaron Cresswell")

    # print("----------2017-2018----------")
    # f = forge.Forge(r'../data/Train_Data/2022-2023_PL_train.csv', model)
    # losses = losses + f.train_one("Aaron Cresswell")

    print("----------2023-2024----------")
    f = forge.Forge(r'../data/Train_Data/2023-2024_PL_train.csv', model, op_fcn=MACROS.optim, l_fcn=MACROS.loss)
    losses = losses + f.train_one("Aaron Cresswell")
except ValueError:
    # do nothing
    print("saving model")


###################################################################################
# If you want to train on all players use
# f.train_all()
# You can also then train to a specific index if you desire
###################################################################################

# plot the model
forge.plot(losses)

# save model
network.save(model, r'Data/model')
