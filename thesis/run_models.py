import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import torch
from torch import nn
import torch.utils.data as data_utils
from thesis.vanilla_nn import MyNeuralNetwork, trainNN, testNN
from thesis.vanilla_rnn import VanillaRNN
from thesis.vanilla_lstm import VanillaLSTM
from thesis.bayes_rnn import BayesRNN
from thesis.bayes_lstm import BayesLSTM
from thesis.utils.markov_sampler import MarkovSamplingLoss

from torch.optim import Adam
from torchmetrics import MeanSquaredError, SpearmanCorrCoef
from tqdm import tqdm, trange

import matplotlib.pyplot as plt


def main():
    ########
    # Hardcoded params
    ########
    data_path = 'C://Users//alexb//Downloads//'
    test_start_date = pd.Timestamp('2021-01-01')
    num_bck_timeframes = 5
    num_weeks_per_bck_timeframe = 2
    # eg. 13 blocks of 4 week returns
    num_fwd_timeframes = 5
    num_weeks_per_fwd_timeframe = 1
    # calculate train end date
    train_end_date = test_start_date - pd.Timedelta(str(int(num_fwd_timeframes*num_weeks_per_fwd_timeframe)+1)+'w')  # make sure no overlap between train and test
    num_samples = 30
    num_epochs = 100
    lr = 0.01

    # num_samples = 10
    # num_epochs = 10

    ########
    # Results tables
    ########
    df_train_rmse = pd.DataFrame()
    df_train_rc = pd.DataFrame()
    df_test_rmse = pd.DataFrame()
    df_test_rc = pd.DataFrame()


    ########
    # Import data and prep dataset
    ########
    df_data, X_train, Y_train, X_test, Y_test, dict_historical_means, scaler_features, scaler_labels = import_and_clean_data(data_path,test_start_date,train_end_date,num_bck_timeframes,num_weeks_per_bck_timeframe,num_fwd_timeframes,num_weeks_per_fwd_timeframe)

    ########
    # Train linear regression
    ########
    # train_linear_regression(df_data,X_train,Y_train,X_test,Y_test,scaler_labels,test_start_date)

    ########
    # "Train" a naive model which uses historical mean
    ########
    # train_naive_model(df_data,Y_test,test_start_date,dict_historical_means,scaler_labels)

    ########
    # Train a standard FF neural network
    ########
    df_train_rmse, df_train_rc, df_test_rmse, df_test_rc = train_vanilla_nn(
        df_data, X_train, Y_train, X_test, Y_test, scaler_labels, test_start_date,train_end_date,num_samples,
        num_epochs,lr,num_fwd_timeframes,df_train_rmse, df_train_rc, df_test_rmse, df_test_rc)

    ########
    # Train a rnn or lstm neural network
    ########
    df_train_rmse, df_train_rc, df_test_rmse, df_test_rc = train_vanilla_rnn_or_lstm(
        df_data, X_train, Y_train, X_test, Y_test, scaler_labels, test_start_date,train_end_date,num_samples,num_epochs,
        lr,'rnn',num_fwd_timeframes,df_train_rmse, df_train_rc, df_test_rmse, df_test_rc)

    df_train_rmse, df_train_rc, df_test_rmse, df_test_rc = train_vanilla_rnn_or_lstm(
        df_data, X_train, Y_train, X_test, Y_test, scaler_labels, test_start_date,train_end_date,num_samples,num_epochs,
        lr,'lstm',num_fwd_timeframes,df_train_rmse, df_train_rc, df_test_rmse, df_test_rc)

    ########
    # Train a Bayesian rnn or lstm neural network
    ########
    df_train_rmse, df_train_rc, df_test_rmse, df_test_rc = train_bayes_rnn_or_lstm(
        df_data, X_train, Y_train, X_test, Y_test, scaler_labels, test_start_date,train_end_date,num_samples,num_epochs,
        lr,'rnn',num_fwd_timeframes,df_train_rmse, df_train_rc, df_test_rmse, df_test_rc,data_path)

    df_train_rmse, df_train_rc, df_test_rmse, df_test_rc = train_bayes_rnn_or_lstm(
        df_data, X_train, Y_train, X_test, Y_test, scaler_labels, test_start_date,train_end_date,num_samples,num_epochs,
        lr,'lstm',num_fwd_timeframes,df_train_rmse, df_train_rc, df_test_rmse, df_test_rc,data_path)

    ########
    # Print LaTeX
    ########
    df_train_rmse.columns = ['Fwd 1 week','Fwd 2 week','Fwd 3 week','Fwd 4 week','Fwd 5 week']
    df_train_rmse.index = ['FFNN', 'RNN','LSTM','Bayes-RNN','Bayes-LSTM']
    df_train_rc.columns = ['Fwd 1 week', 'Fwd 2 week', 'Fwd 3 week', 'Fwd 4 week', 'Fwd 5 week']
    df_train_rc.index = ['FFNN', 'RNN','LSTM','Bayes-RNN','Bayes-LSTM']
    df_test_rmse.columns = ['Fwd 1 week', 'Fwd 2 week', 'Fwd 3 week', 'Fwd 4 week', 'Fwd 5 week']
    df_test_rmse.index = ['FFNN', 'RNN','LSTM','Bayes-RNN','Bayes-LSTM']
    df_test_rc.columns = ['Fwd 1 week', 'Fwd 2 week', 'Fwd 3 week', 'Fwd 4 week', 'Fwd 5 week']
    df_test_rc.index = ['FFNN', 'RNN','LSTM','Bayes-RNN','Bayes-LSTM']

    print(df_train_rmse.to_latex(position='t',column_format='|lccccc|').replace('\\\\','\\\\ \\hline').replace('toprule','hline'))
    print(df_train_rc.to_latex(position='t',column_format='|lccccc|').replace('\\\\','\\\\ \\hline').replace('toprule','hline'))
    print(df_test_rmse.to_latex(position='t',column_format='|lccccc|').replace('\\\\','\\\\ \\hline').replace('toprule','hline'))
    print(df_test_rc.to_latex(position='t',column_format='|lccccc|').replace('\\\\','\\\\ \\hline').replace('toprule','hline'))


def import_and_clean_data(data_path,test_start_date,train_end_date,num_bck_timeframes,num_weeks_per_bck_timeframe,num_fwd_timeframes,num_weeks_per_fwd_timeframe):
    l_instr = ['sp500','nasdaq','asx200','nikkei225','dax','eurostoxx50']

    # use nasdaq for date index, sp500 has data issues where the week end on Thursdays on random weeks, but order of
    # weekly returns is still correct
    df_date = pd.read_csv(data_path+'nasdaq'+'.csv')
    df_date['date_time'] = pd.to_datetime(df_date['date_time'], format="%d/%m/%Y")
    # import data
    df_data = pd.DataFrame()
    for instr in l_instr:
        df_ = pd.read_csv(data_path+instr+'.csv')
        df_ = df_.drop('date_time',axis=1)
        df_['date_time'] = df_date['date_time']  # use the nasdaq date index, from above
        df_['instr'] = instr
        if len(df_data)==0:
            df_data = df_.copy()
        else:
            df_data = df_data.append(df_,sort=False)
    # cull data after February 2023
    df_data = df_data[df_data['date_time']<pd.Timestamp("2023-03-01")].copy()
    # define column order
    df_data = df_data[['date_time','instr','close']]

    # calculate backward returns: 52 weeks
    x_cols = []
    for i in range(0,num_bck_timeframes):
        df_data['bck_'+str(i)] = df_data.groupby('instr')['close'].shift(i*num_weeks_per_bck_timeframe) / df_data.groupby('instr')['close'].shift((i+1)*num_weeks_per_bck_timeframe) - 1
        x_cols.append('bck_'+str(i))
        # add a squared 4 week backward return feature
        # if i == 0:
        #     df_data['bck_sq_'+str(i)] = (df_data.groupby('instr')['close'].shift(i*num_weeks_per_bck_timeframe) / df_data.groupby('instr')['close'].shift((i+1)*num_weeks_per_bck_timeframe) - 1)**2
        #     x_cols.append('bck_sq_'+str(i))

    # calculate label: forward 4 week return
    y_cols = []
    for i in range(1, num_fwd_timeframes + 1):
        df_data['label_' + str(i)] = df_data.groupby('instr')['close'].shift(-i*num_weeks_per_fwd_timeframe) / df_data['close'] - 1
        y_cols.append('label_' + str(i))
    # drop rows with nans
    df_data = df_data.dropna()
    # capture historical mean for use in a naive model later
    dict_historical_means = df_data[df_data['date_time']<=train_end_date].groupby('instr')['label_'+str(i)].mean().to_dict()

    # features: pivot
    # df_melt = pd.melt(df_data, id_vars=['date_time','instr'], value_vars=x_cols)
    # df_pivot = df_melt.pivot(index='date_time', columns=['instr','variable'], values='value')
    # df_pivot = df_pivot.reset_index()
    # df_pivot.columns.name = None
    # df_pivot.columns = df_pivot.columns.map('_'.join).str.strip('_')

    # drop x_cols, and then merge data with df_pivot
    # df_data = df_data.drop(x_cols,axis=1)
    # df_data = df_data.merge(df_pivot,how='left',on='date_time')
    # # calculate new list of x_col
    # x_cols = df_pivot.columns.to_list()
    # x_cols = x_cols[1:].copy()  # don't keep the 'date_time' col in list of x_cols

    # split data into train and test
    # we have data from 2011 to 2022: use 2021+ for test
    X_train = df_data[df_data['date_time'] <= train_end_date][x_cols]
    Y_train = df_data[df_data['date_time'] <= train_end_date][y_cols]
    X_test = df_data[df_data['date_time'] >= test_start_date][x_cols]
    Y_test = df_data[df_data['date_time'] >= test_start_date][y_cols]

    # minmax scale the data
    scaler_features = StandardScaler()
    scaler_labels = StandardScaler()  # for inverting -1:1 preds back to returns
    scaler_features.fit(X_train)
    X_train = scaler_features.transform(X_train)
    X_test = scaler_features.transform(X_test)
    scaler_labels.fit(Y_train.values)
    Y_train = scaler_labels.transform(Y_train.values)

    # clip very large outliers for training
    Y_train = np.clip(Y_train, -2, 2)
    Y_train = Y_train - np.mean(Y_train,axis=0)

    return df_data, X_train, Y_train, X_test, Y_test, dict_historical_means, scaler_features, scaler_labels


def train_linear_regression(df_data,X_train,Y_train,X_test,Y_test,scaler_labels,test_start_date):
    linreg = LinearRegression()
    linreg.fit(X_train, Y_train.values.ravel())
    # make preds
    linreg_preds = linreg.predict(X_test)
    linreg_preds_returns = scaler_labels.inverse_transform(linreg_preds.reshape(-1, 1))
    Y_test_returns = scaler_labels.inverse_transform(Y_test.values.reshape(-1, 1))
    # evaluate preds
    rmse = mean_squared_error(Y_test_returns, linreg_preds_returns) ** 0.5
    r_squared = r2_score(Y_test_returns, linreg_preds_returns)
    # rank correlations
    df_test = df_data[df_data['date_time'] >= test_start_date].copy()
    df_test = df_test.reset_index(level=0, drop=True)
    df_test['label_rank'] = df_test.groupby('date_time')['label'].rank(ascending=True).reset_index(level=0, drop=True)
    df_test['linreg_preds'] = linreg_preds_returns.flatten()
    df_test['linreg_preds_rank'] = df_test.groupby('date_time')['linreg_preds'].rank(ascending=True).reset_index(
        level=0, drop=True)
    # print correlation of ranks
    rank_correls = round(np.corrcoef(df_test.dropna()['label_rank'], df_test.dropna()['linreg_preds_rank'])[0][1], 4)
    print('linreg rmse, exp_var, rank_correls:', round(rmse, 4), round(r_squared, 4), rank_correls)


def train_naive_model(df_data,Y_test,test_start_date,dict_historical_means,scaler_labels):
    naive_preds_returns = df_data[df_data['date_time'] >= test_start_date]['instr'].map(dict_historical_means)
    Y_test_returns = scaler_labels.inverse_transform(Y_test.values.reshape(-1, 1))
    rmse = mean_squared_error(Y_test_returns, naive_preds_returns) ** 0.5
    r_squared = r2_score(Y_test_returns, naive_preds_returns)
    df_test = df_data[df_data['date_time'] >= test_start_date].copy()
    df_test = df_test.reset_index(level=0, drop=True)
    df_test['label_rank'] = df_test.groupby('date_time')['label'].rank(ascending=True).reset_index(level=0, drop=True)
    df_test['naive_preds'] = naive_preds_returns.reset_index(level=0, drop=True)
    df_test['naive_preds_rank'] = df_test.groupby('date_time')['naive_preds'].rank(ascending=True).reset_index(level=0,drop=True)
    # print correlation of ranks
    rank_correls = round(np.corrcoef(df_test.dropna()['label_rank'], df_test.dropna()['naive_preds_rank'])[0][1], 4)
    print('naive rmse, exp_var, rank_correls:', round(rmse, 4), round(r_squared, 4), rank_correls)


def train_vanilla_nn(df_data, X_train, Y_train, X_test, Y_test, scaler_labels, test_start_date,train_end_date,
                     num_samples,num_epochs,lr,num_fwd_timeframes,df_train_rmse, df_train_rc, df_test_rmse, df_test_rc):

    # Passing to DataLoader
    batch_size = 111
    train_tensor = data_utils.TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float())
    train_loader = data_utils.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_tensor = data_utils.TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test.values).float())
    test_loader = data_utils.DataLoader(test_tensor, batch_size=batch_size, shuffle=True)

    device = 'cpu'
    models = [
        MyNeuralNetwork(output_dim=num_fwd_timeframes).to(device)
        for s in range(num_samples)
    ]

    # loss fn and optimiser
    loss_fn = nn.MSELoss()

    # MSE
    train_rmse_list = []
    train_rankcorrels_list = []
    test_rmse_list = []
    test_rankcorrels_list = []

    for sample in range(0, num_samples):

        # Model to train
        model = models[sample]

        # Optimizer
        optimizer = Adam(model.parameters(), lr=lr)

        print(f"Model: {model.__class__.__name__} Sample: {sample}")

        for t in range(num_epochs):
            # print(f"Epoch {t+1}\n-------------------------------")
            trainNN(train_loader, model, loss_fn, optimizer)
            testNN(test_loader, model, loss_fn,t)

        # output preds
        nn_preds_train = model(torch.tensor(X_train).float())
        nn_preds_train = nn_preds_train.cpu().detach().numpy()
        nn_preds = model(torch.tensor(X_test).float())
        nn_preds = nn_preds.cpu().detach().numpy()

        # evaluate preds for this sample
        test_rmse_list, test_rankcorrels_list, train_rmse_list, train_rankcorrels_list = \
            calculate_performance_this_sample(scaler_labels, nn_preds, nn_preds_train, Y_test, Y_train,
                                              num_fwd_timeframes, df_data,
                                              test_start_date, train_end_date, test_rmse_list, test_rankcorrels_list,
                                              train_rmse_list, train_rankcorrels_list)

    # calculate mean and std dev of each metric, across the samples
    l1,l2,l3,l4 = calculate_performance_across_samples(num_fwd_timeframes, train_rmse_list, train_rankcorrels_list,
                                                       test_rmse_list, test_rankcorrels_list, "", "Vanilla FFNN")

    # update dataframes with resuls for this model
    df_train_rmse, df_train_rc, df_test_rmse, df_test_rc = update_results_df(l1, l2, l3, l4,
                                                                             df_train_rmse, df_train_rc,
                                                                             df_test_rmse, df_test_rc)

    return df_train_rmse, df_train_rc, df_test_rmse, df_test_rc


def train_vanilla_rnn_or_lstm(df_data, X_train, Y_train, X_test, Y_test, scaler_labels, test_start_date,train_end_date,
                              num_samples,num_epochs,lr,model_type,num_fwd_timeframes,df_train_rmse, df_train_rc, df_test_rmse, df_test_rc):

    # Passing to DataLoader
    batch_size = 111
    train_tensor = data_utils.TensorDataset(torch.tensor(np.expand_dims(X_train, axis=2)).float(), torch.tensor(Y_train).float())
    train_loader = data_utils.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_tensor = data_utils.TensorDataset(torch.tensor(np.expand_dims(X_test, axis=2)).float(), torch.tensor(Y_test.values).float())
    test_loader = data_utils.DataLoader(test_tensor, batch_size=batch_size, shuffle=True)

    # Dimensions
    input_dim = 1
    hidden_dim = 32
    output_dim = num_fwd_timeframes

    # Model
    if model_type == 'rnn':
        models = [
            VanillaRNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
            for s in range(num_samples)
        ]
    elif model_type == 'lstm':
        models = [
            VanillaLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
            for s in range(num_samples)
        ]

    #########
    # train #
    #########

    # MSE Metric
    metric = MeanSquaredError()
    mse_loss = nn.MSELoss()

    # MSE
    train_rmse_list = []
    train_rankcorrels_list = []
    test_rmse_list = []
    test_rankcorrels_list = []

    for sample in range(0, num_samples):

        # Model to train
        model = models[sample]

        # Optimizer
        optimizer = Adam(model.parameters(), lr=lr)

        print(f"Model: {model.__class__.__name__} Sample: {sample}")

        # Sampler
        model.init_weights()
        model.train()

        # Progress bar
        pbar = trange(num_epochs, desc=f"Training {model.__class__.__name__}")

        for epoch in pbar:

            # Reset metric
            metric.reset()
            epoch_loss = 0

            for X, y in train_loader:
                # Reset the gradients
                model.zero_grad()

                # Predict and compute loss
                output = model(X)
                loss = mse_loss(output, y)

                # Backpropagation
                loss.backward()
                optimizer.step()

                # Update metric
                metric.update(output, y)

                # Total loss
                epoch_loss += loss.detach().numpy()

            # Validation
            test_rmse = evaluate_vanilla(model, test_loader)
            train_rmse = torch.sqrt(metric.compute())

            # Update the progress bar
            pbar.set_description(
                f"Train Loss: {epoch_loss: .4f} RMSE: {train_rmse:.4f} Test RMSE: {test_rmse:.4f}"
            )

        # output preds
        nn_preds_train = model(torch.tensor(np.expand_dims(X_train, axis=2)).float())
        nn_preds_train = nn_preds_train.cpu().detach().numpy()
        nn_preds = model(torch.tensor(np.expand_dims(X_test, axis=2)).float())
        nn_preds = nn_preds.cpu().detach().numpy()
        print("mean of preds", np.mean(nn_preds))
        print("std dev of preds", np.std(nn_preds))

        # evaluate preds for this sample
        test_rmse_list, test_rankcorrels_list, train_rmse_list, train_rankcorrels_list = \
            calculate_performance_this_sample(scaler_labels, nn_preds, nn_preds_train, Y_test, Y_train,
                                              num_fwd_timeframes, df_data,
                                              test_start_date, train_end_date, test_rmse_list, test_rankcorrels_list,
                                              train_rmse_list, train_rankcorrels_list)

    # calculate mean and std dev of each metric, across the samples
    l1, l2, l3, l4 = calculate_performance_across_samples(num_fwd_timeframes, train_rmse_list, train_rankcorrels_list,
                                                          test_rmse_list, test_rankcorrels_list, model_type, "Vanilla")

    # update dataframes with resuls for this model
    df_train_rmse, df_train_rc, df_test_rmse, df_test_rc = update_results_df(l1, l2, l3, l4,
                                                                             df_train_rmse, df_train_rc,
                                                                             df_test_rmse, df_test_rc)

    return df_train_rmse, df_train_rc, df_test_rmse, df_test_rc


def train_bayes_rnn_or_lstm(df_data, X_train, Y_train, X_test, Y_test, scaler_labels, test_start_date,train_end_date,
                            num_samples,num_epochs,lr,model_type,num_fwd_timeframes,df_train_rmse, df_train_rc,
                            df_test_rmse, df_test_rc,data_path):

    # Passing to DataLoader
    batch_size = 111
    train_tensor = data_utils.TensorDataset(torch.tensor(np.expand_dims(X_train, axis=2)).float(), torch.tensor(Y_train).float())
    train_loader = data_utils.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    # Dimensions
    input_dim = 1
    hidden_dim = 32
    output_dim = num_fwd_timeframes

    # Model
    if model_type == 'rnn':
        model = BayesRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim)
    elif model_type == 'lstm':
        model = BayesLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim)

    #########
    # train #
    #########

    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    # MSE Metric
    metric = MeanSquaredError()
    # Number of batches
    num_batches = len(train_loader)
    # Progress bar
    pbar = trange(num_epochs, desc=f"Training {model.__class__.__name__}")

    # Sampler
    model.train()
    sampling_loss = MarkovSamplingLoss(model, samples=num_samples)


    for epoch in pbar:

        # Reset metric
        metric.reset()
        epoch_loss = 0

        for X, y in train_loader:

            # Reset the gradients
            model.zero_grad()

            # Compute sampling loss
            loss, outputs = sampling_loss(X, y, num_batches)

            # Backpropagation
            loss.backward(retain_graph=True)
            optimizer.step()

            # Update metric
            metric.update(outputs.mean(dim=0), y)

            # Total loss
            epoch_loss += loss.detach().numpy()

        # Validation
        train_rmse = torch.sqrt(metric.compute())

        # Update the progress bar
        pbar.set_description(
            f"Train Loss: {epoch_loss: .4f} RMSE: {train_rmse:.4f}"
        )

    # we have now finished training
    model.training = False

    # MSE
    train_rmse_list = []
    train_rankcorrels_list = []
    test_rmse_list = []
    test_rankcorrels_list = []

    ###########################################
    # look at mean and std dev of predictions #
    ###########################################
    nn_preds = model(torch.tensor(np.expand_dims(X_test, axis=2)).float()).cpu().detach().numpy()
    # print("mean of preds", np.mean(nn_preds,axis=0))
    # print("std dev of preds",np.std(nn_preds,axis=0))
    nn_preds_returns = scaler_labels.inverse_transform(nn_preds)
    print("mean of preds", np.mean(nn_preds_returns, axis=0))
    print("std dev of preds", np.std(nn_preds_returns, axis=0))
    print("mean of test_y", np.mean(Y_test, axis=0))
    print("std dev of test_y", np.std(Y_test, axis=0))

    #########
    # plots #
    #########
    df_test = df_data[df_data['date_time'] >= test_start_date].copy()
    df_test = df_test.reset_index(level=0, drop=True)
    for i in range(1,nn_preds_returns.shape[1]+1):
        df_test['pred'+str(i)] = nn_preds_returns[:,i-1]

    for u in range(0,nn_preds_returns.shape[1]):
        y_pred_list = []
        for i in range(100):
            y_pred = model(torch.tensor(np.expand_dims(X_test, axis=2)).float(),sampling=True).cpu().detach().numpy()[:,u]
            y_pred_list.append(y_pred)
        y_mean = np.expand_dims(np.mean(y_pred_list, axis=0),axis=1)
        y_sigma = np.expand_dims(np.std(y_pred_list, axis=0),axis=1)
        if u == 0:
            y_means = y_mean.copy()
            y_sigmas = y_sigma.copy()
        else:
            y_means = np.concatenate((y_means, y_mean),axis=1)
            y_sigmas = np.concatenate((y_sigmas, y_sigma), axis=1)

    y_sigmas = scaler_labels.inverse_transform(y_sigmas)
    for i in range(1,nn_preds_returns.shape[1]+1):
        df_test['std'+str(i)] = y_sigmas[:,i-1]

    for i in range(1,nn_preds_returns.shape[1]+1):
        df_test['pred_close'+str(i)] = df_test['close']*(1+df_test['pred'+str(i)])
        df_test['pred_close_lb' + str(i)] = df_test['close'] * (1 - df_test['std' + str(i)])
        df_test['pred_close_ub' + str(i)] = df_test['close'] * (1 + df_test['std' + str(i)])

    # 3 line charts
    dict_ymax = {'sp500':4600,'nasdaq':15500, 'asx200':7700, 'nikkei225':31000, 'dax':18000, 'eurostoxx50':4400}
    for instr_ in ['sp500', 'nasdaq', 'asx200', 'nikkei225', 'dax', 'eurostoxx50']:
        f, a = plt.subplots(1, 3,figsize=(10, 4))
        a = a.ravel()
        nums = [20,30,40]
        for idx, ax in enumerate(a):
            num = nums[idx]
            df_ = df_test[(df_test['instr']==instr_)].iloc[:num,:].copy()
            df_ = df_.reset_index(drop=True)
            df_['pred_line'] = (num-6)*[np.nan]+[df_['close'].loc[num-6],df_['pred_close1'].loc[num-6],df_['pred_close2'].loc[num-6],df_['pred_close3'].loc[num-6],df_['pred_close4'].loc[num-6],df_['pred_close5'].loc[num-6]]
            df_['pred_line_low'] = (num-6)*[np.nan]+[df_['close'].loc[num-6],df_['pred_close_lb1'].loc[num-6],df_['pred_close_lb2'].loc[num-6],df_['pred_close_lb3'].loc[num-6],df_['pred_close_lb4'].loc[num-6],df_['pred_close_lb5'].loc[num-6]]
            df_['pred_line_high'] = (num-6)*[np.nan]+[df_['close'].loc[num-6],df_['pred_close_ub1'].loc[num-6],df_['pred_close_ub2'].loc[num-6],df_['pred_close_ub3'].loc[num-6],df_['pred_close_ub4'].loc[num-6],df_['pred_close_ub5'].loc[num-6]]
            ax.plot(df_.index, df_['pred_line'], label='Prediction')
            ax.plot(df_.index, df_['close'], label='Actual')
            ax.plot(df_.index, df_['pred_line_low'], label='Uncertainty', color='blue', linestyle='--', alpha=0.25)
            ax.plot(df_.index, df_['pred_line_high'], color='blue', linestyle='--', alpha=0.25)
            ax.fill_between(df_.index, df_['pred_line_low'], df_['pred_line_high'], facecolor='grey', alpha=0.2)
            ax.set_xlim(xmax=45)
            ax.set_ylim(ymax=dict_ymax[instr_])
            ax.set_xlabel('Time (weeks)')
            if idx == 0:
                ax.legend(loc='upper left')
                ax.set_ylabel('Closing price')
            if idx != 0:
                ax.set_yticks([])
        plt.tight_layout(pad=1.0)
        plt.savefig(data_path + 'progression_'+instr_+'.png')

    # df_test['actual5'] = df_test.groupby('instr')['close'].shift(-5)
    # df_test['pred_close5'] = df_test['close']*(1+df_test['pred5'])
    # df_ = df_test[(df_test['instr']=='sp500') & (df_test['actual5'].isnull() == False)].iloc[::5,:].copy()
    # df_['cumprod_pred5'] = (1+df_['pred5']).cumprod()
    # df_['pred_cum_close5'] = df_['close']*df_['cumprod_pred5']
    # df_['pred_cum_close5_lowerbound'] = df_['pred_close5']*(1-df_['std5'])
    # df_['pred_cum_close5_lupperbound'] = df_['pred_close5'] * (1 + df_['std5'])
    #
    # plt.plot(df_['date_time'], df_['pred_close5'], label='Prediction')
    # plt.plot(df_['date_time'], df_['actual5'], label='Actual')
    # plt.plot(df_['date_time'], df_['pred_cum_close5_lowerbound'], label='Uncertainty', color='blue', linestyle='--', alpha=0.25)
    # plt.plot(df_['date_time'], df_['pred_cum_close5_lupperbound'], color='blue', linestyle='--', alpha=0.25)
    # plt.fill_between(df_['date_time'], df_['pred_cum_close5_lowerbound'], df_['pred_cum_close5_lupperbound'], facecolor='grey', alpha=0.2)
    #
    # df_test['actual1'] = df_test.groupby('instr')['close'].shift(-1)
    # df_test['pred_close1'] = df_test['close']*(1+df_test['pred1'])
    # df_ = df_test[(df_test['instr']=='asx200') & (df_test['actual1'].isnull() == False)].iloc[-30:,:].copy()
    # df_['cumprod_pred1'] = (1+df_['pred1']).cumprod()
    # df_['pred_cum_close1'] = df_['close']*df_['cumprod_pred1']
    # df_['pred_cum_close1_lowerbound'] = df_['pred_close1']*(1-df_['std1'])
    # df_['pred_cum_close1_lupperbound'] = df_['pred_close1'] * (1 + df_['std1'])
    #
    # plt.plot(df_['date_time'], df_['pred_close1'], label='Prediction')
    # plt.plot(df_['date_time'], df_['actual1'], label='Actual')
    # plt.plot(df_['date_time'], df_['pred_cum_close1_lowerbound'], label='Uncertainty', color='blue', linestyle='--', alpha=0.25)
    # plt.plot(df_['date_time'], df_['pred_cum_close1_lupperbound'], color='blue', linestyle='--', alpha=0.25)
    # plt.fill_between(df_['date_time'], df_['pred_cum_close1_lowerbound'], df_['pred_cum_close1_lupperbound'], facecolor='grey', alpha=0.2)

    for sample in range(0, num_samples):

        # output preds, don't sample of the first set.
        if sample == 0:
            nn_preds_train = model(torch.tensor(np.expand_dims(X_train, axis=2)).float())
            nn_preds_train = nn_preds_train.cpu().detach().numpy()
            nn_preds = model(torch.tensor(np.expand_dims(X_test, axis=2)).float())
            nn_preds = nn_preds.cpu().detach().numpy()
        else:
            nn_preds_train = model(torch.tensor(np.expand_dims(X_train, axis=2)).float(), sampling=True)
            nn_preds_train = nn_preds_train.cpu().detach().numpy()
            nn_preds = model(torch.tensor(np.expand_dims(X_test, axis=2)).float(), sampling=True)
            nn_preds = nn_preds.cpu().detach().numpy()

        # evaluate preds for this sample
        test_rmse_list, test_rankcorrels_list, train_rmse_list, train_rankcorrels_list = \
            calculate_performance_this_sample(scaler_labels, nn_preds, nn_preds_train, Y_test, Y_train,
                                              num_fwd_timeframes, df_data,
                                              test_start_date, train_end_date, test_rmse_list, test_rankcorrels_list,
                                              train_rmse_list, train_rankcorrels_list)

    # calculate mean and std dev of each metric, across the samples
    l1,l2,l3,l4 = calculate_performance_across_samples(num_fwd_timeframes, train_rmse_list, train_rankcorrels_list,
                                                       test_rmse_list, test_rankcorrels_list, model_type, "Bayes")

    # update dataframes with resuls for this model
    df_train_rmse, df_train_rc, df_test_rmse, df_test_rc = update_results_df(l1, l2, l3, l4,
                                                                             df_train_rmse, df_train_rc,
                                                                             df_test_rmse, df_test_rc)

    return df_train_rmse, df_train_rc, df_test_rmse, df_test_rc


def evaluate_vanilla(model, dataloader):
    model.eval()
    # MSE Metric
    metric = MeanSquaredError()
    for batch, (X, y) in enumerate(dataloader):
        # Output
        out = model(X)
        # Update metric
        metric.update(out, y)
    # Compute RMSE
    rmse = torch.sqrt(metric.compute())
    return rmse


def update_results_df(l1, l2, l3, l4, df_train_rmse, df_train_rc, df_test_rmse, df_test_rc):
    df_train_rmse = df_train_rmse.append(pd.DataFrame(l1).transpose())
    df_train_rc = df_train_rc.append(pd.DataFrame(l2).transpose())
    df_test_rmse = df_test_rmse.append(pd.DataFrame(l3).transpose())
    df_test_rc = df_test_rc.append(pd.DataFrame(l4).transpose())

    return df_train_rmse, df_train_rc, df_test_rmse, df_test_rc


def calculate_performance_this_sample(scaler_labels,nn_preds,nn_preds_train,Y_test,Y_train,num_fwd_timeframes,df_data,
                                      test_start_date,train_end_date,test_rmse_list, test_rankcorrels_list,
                                      train_rmse_list, train_rankcorrels_list):
    # evaluate preds
    nn_preds_returns = scaler_labels.inverse_transform(nn_preds)
    nn_preds_train_returns = scaler_labels.inverse_transform(nn_preds_train)
    # evaluate preds
    Y_test_returns = Y_test.values
    Y_train_returns = scaler_labels.inverse_transform(Y_train)
    rmse = []
    rmse_train = []
    for i in range(0,num_fwd_timeframes):
        rmse.append(round(mean_squared_error(Y_test_returns[:,i], nn_preds_returns[:,i])**0.5,4))
        rmse_train.append(round(mean_squared_error(Y_train_returns[:, i], nn_preds_train_returns[:, i]) ** 0.5, 4))
    df_test = df_data[df_data['date_time'] >= test_start_date].copy()
    df_test = df_test.reset_index(level=0, drop=True)
    df_train = df_data[df_data['date_time'] <= train_end_date].copy()
    df_train = df_train.reset_index(level=0, drop=True)
    rank_correls = []
    rank_correls_train = []
    for i in range(0,num_fwd_timeframes):
        df_test['label_rank'] = df_test.groupby('date_time')['label_'+str(i+1)].rank(ascending=True).reset_index(level=0, drop=True)
        df_test['nn_preds'] = nn_preds_returns[:,i]
        df_test['nn_preds_rank'] = df_test.groupby('date_time')['nn_preds'].rank(ascending=True).reset_index(level=0,drop=True)
        df_train['label_rank'] = df_train.groupby('date_time')['label_'+str(i+1)].rank(ascending=True).reset_index(level=0, drop=True)
        df_train['nn_preds'] = nn_preds_train_returns[:,i]
        df_train['nn_preds_rank'] = df_train.groupby('date_time')['nn_preds'].rank(ascending=True).reset_index(level=0,drop=True)
        rank_correls.append(round(np.corrcoef(df_test.dropna()['label_rank'], df_test.dropna()['nn_preds_rank'])[0][1],4))
        rank_correls_train.append(round(np.corrcoef(df_train.dropna()['label_rank'], df_train.dropna()['nn_preds_rank'])[0][1],4))
    # print('nn rmse, rank_correls:',rmse,rank_correls)

    # Append MSE
    test_rmse_list.append(rmse)
    test_rankcorrels_list.append(rank_correls)
    train_rmse_list.append(rmse_train)
    train_rankcorrels_list.append(rank_correls_train)

    return test_rmse_list, test_rankcorrels_list, train_rmse_list, train_rankcorrels_list


def calculate_performance_across_samples(num_fwd_timeframes,train_rmse_list,train_rankcorrels_list,test_rmse_list,
                                         test_rankcorrels_list,model_type,model_name):
    train_rmse_mean = []
    train_rmse_std = []
    train_rc_mean = []
    train_rc_std = []
    test_rmse_mean = []
    test_rmse_std = []
    test_rc_mean = []
    test_rc_std = []
    for i in range(0, num_fwd_timeframes):
        train_rmse_mean.append(round(np.array(train_rmse_list)[0, i], 4))
        train_rmse_std.append(round(np.array(train_rmse_list)[:, i].std(), 4))
        train_rc_mean.append(round(np.array(train_rankcorrels_list)[0, i], 4))
        train_rc_std.append(round(np.array(train_rankcorrels_list)[:, i].std(), 4))
        test_rmse_mean.append(round(np.array(test_rmse_list)[0, i], 4))
        test_rmse_std.append(round(np.array(test_rmse_list)[:, i].std(), 4))
        test_rc_mean.append(round(np.array(test_rankcorrels_list)[0, i], 4))
        test_rc_std.append(round(np.array(test_rankcorrels_list)[:, i].std(), 4))

    print("------across all samples, "+model_name, model_type, ": train RMSE, mean:", train_rmse_mean)
    print("------across all samples, "+model_name, model_type, ": train RMSE, std:", train_rmse_std)
    print("------across all samples, "+model_name, model_type, ": train RC, mean:", train_rc_mean)
    print("------across all samples, "+model_name, model_type, ": train RC, std:", train_rc_std)
    print("------across all samples, "+model_name, model_type, ": test RMSE, mean:", test_rmse_mean)
    print("------across all samples, "+model_name, model_type, ": test RMSE, std:", test_rmse_std)
    print("------across all samples, "+model_name, model_type, ": test RC, mean:", test_rc_mean)
    print("------across all samples, "+model_name, model_type, ": test RC, std:", test_rc_std)

    l1 = []
    l2 = []
    l3 = []
    l4 = []
    for i in range(0, len(train_rmse_mean)):
        l1.append(format(train_rmse_mean[i], '.4f') + ' (+/-' + format(train_rmse_std[i], '.3f') + ')')
        l2.append(format(train_rc_mean[i], '.4f') + ' (+/-' + format(train_rc_std[i], '.3f') + ')')
        l3.append(format(test_rmse_mean[i], '.4f') + ' (+/-' + format(test_rmse_std[i], '.3f') + ')')
        l4.append(format(test_rc_mean[i], '.4f') + ' (+/-' + format(test_rc_std[i], '.3f') + ')')

    return l1, l2, l3, l4


if __name__ == "__main__":
    main()
