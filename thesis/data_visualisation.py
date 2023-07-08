import pandas as pd
import numpy as np
from thesis.run_models import import_and_clean_data
import matplotlib.pyplot as plt

########
# Hardcoded params
########
data_path = 'C://Users//alexb//Downloads//'
test_start_date = pd.Timestamp('2021-01-01')
num_bck_timeframes = 13
num_weeks_per_bck_timeframe = 4
# eg. 13 blocks of 4 week returns
num_fwd_timeframes = 5
num_weeks_per_fwd_timeframe = 1
# calculate train end date
train_end_date = test_start_date - pd.Timedelta(str(int(num_fwd_timeframes*num_weeks_per_fwd_timeframe)+1)+'w')  # make sure no overlap between train and test

###############
# import data #
###############
df_data, X_train, Y_train, X_test, Y_test, dict_historical_means, scaler_features, scaler_labels = import_and_clean_data(data_path,test_start_date,train_end_date,num_bck_timeframes,num_weeks_per_bck_timeframe,num_fwd_timeframes,num_weeks_per_fwd_timeframe)
df_data = df_data[['date_time','instr','close']].copy()


########
# TRIs #
########
# pivot
df_pivot = df_data.pivot(index='date_time', columns=['instr'], values='close')
df_pivot = df_pivot.reset_index()
df_pivot.columns.name = None
# rebase all the prices to start at 1
cols = df_pivot.columns.to_list()
plt.figure(1)
for i in range(1,len(cols)):
    instr_ = cols[i]
    starting_value = df_pivot.at[0,instr_]
    df_pivot[instr_] = df_pivot[instr_]/starting_value
    # plot
    plt.plot(df_pivot['date_time'],df_pivot[instr_],linewidth=0.75)
plt.legend(['S&P/ASX 200','DAX','Euro Stoxx 50','Nasdaq-100','Nikkei 225','S&P 500'])
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
plt.savefig(data_path+'TRI.png')

##############
# Histograms #
##############
data_list = []
for i in range(1,len(cols)):
    instr_ = cols[i]
    starting_value = df_pivot.at[0,instr_]
    df_pivot[instr_] = df_pivot[instr_]/df_pivot[instr_].shift(1) - 1
    data_list.append(df_pivot[instr_].dropna().to_list())
flat_list = [item for sublist in data_list for item in sublist]
data_list.insert(0,flat_list)
data_list.insert(1,[])
# 7 histograms
xaxes = ['','','','','','','Weekly % return','Weekly % return']
yaxes = ['Frequency','','Frequency','','Frequency','','Frequency','']
titles = ['Global','','S&P/ASX 200', 'DAX', 'Euro Stoxx 50', 'Nasdaq-100', 'Nikkei 225', 'S&P 500']
f,a = plt.subplots(4,2)
a = a.ravel()
for idx,ax in enumerate(a):
    ax.hist(data_list[idx], bins=30)
    ax.set_title(titles[idx])
    ax.set_xlabel(xaxes[idx])
    ax.set_ylabel(yaxes[idx])
    ax.set_xlim(-0.15,0.15)
f.delaxes(a[1]) # delete the unused plot
plt.tight_layout(pad=1.0)
plt.savefig(data_path+'histograms.png')

# global histogram
flat_list = [item for sublist in data_list for item in sublist]
plt.figure(3)
plt.hist(flat_list,bins=30)
plt.title(label="Global")
plt.show()
plt.savefig('histograms_1.png')

# ######################
# # Prediction problem #
# ######################
# X_train_returns = scaler_features.inverse_transform(X_train.values)
# Y_train_returns = scaler_labels.inverse_transform(Y_train.values)
# df_X = pd.DataFrame(X_train_returns)
# df_Y = pd.DataFrame(Y_train_returns)
# df_X.iloc[[0,200,400,600,800,1000,1200,1400,1600,1800]].to_clipboard()
# df_Y.iloc[[0,200,400,600,800,1000,1200,1400,1600,1800]].to_clipboard()