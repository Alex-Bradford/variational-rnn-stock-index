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
# Bar charts #
##############
# Build the plot
# x points
N = 5
x = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars
# data
train_rmse = [[0.0215 ,0.0289 ,0.0349 ,0.0397 ,0.0441],[0.0218,0.0291 ,0.0350 ,0.0403 ,0.0447],
              [0.0201 ,0.0270 ,0.0324 ,0.0369 ,0.0414],[0.0213 ,0.0286 ,0.0340 ,0.0390 ,0.0435],
              [0.0215 ,0.0292 ,0.0353 ,0.0406 ,0.0454]]
train_rc = [[0.0660,0.0850 ,0.1044 ,0.1085 ,0.0830],[0.0439,-0.0245,0.0184,-0.0480,-0.0362],
            [0.1021,0.1340,0.1734,0.1907,0.1657],[0.0832,0.0932,0.1454, 0.1333,0.1120],
            [0.0389,0.0433,0.0258,0.0318,0.0363]]
test_rmse = [[0.0254,0.0356,0.0433,0.0494,0.0540],[0.0255,0.0357,0.0431,0.0495,0.0537],
            [0.0264,0.0376,0.0461,0.0525,0.0580],[0.0253,0.0362,0.0444,0.0500,0.0545],
            [0.0253,0.0354,0.0433,0.0498,0.0548]]
test_rc = [[0.0339,-0.0185 ,0.0005,0.0085,0.0217],[0.0011,0.0397,0.0180,0.0116,0.0767],
        [0.0254,0.0280,0.0090,0.0384,0.0878],[0.0847,0.0910,0.1074,0.1132,0.0984],
        [0.0429,0.0492,0.0159,-0.0122,-0.0524]]
data = [train_rmse,train_rc,test_rmse,test_rc]
e_train_rmse = [[0.00005, 0.0001, 0.0001, 0.0001, 0.0001],[0.0012, 0.0004, 0.0005, 0.0006, 0.0007],[0.0001, 0.0002, 0.0002, 0.0003, 0.0003],[0.0003, 0.0005, 0.0006, 0.0008, 0.0005],[0.00005, 0.0001, 0.0001, 0.0001, 0.0001]]
e_train_rc = [[0.0127, 0.0129, 0.0139, 0.0141, 0.0155],[0.0316, 0.0385, 0.0415, 0.0409, 0.0415],[0.0184, 0.0191, 0.0209, 0.0219, 0.0215],[0.0178, 0.0116, 0.0143, 0.013, 0.0148],[0.0141, 0.0128, 0.0135, 0.0103, 0.0133]]
e_test_rmse = [[0.0001, 0.0002, 0.0003, 0.0004, 0.0006],[0.0012, 0.0006, 0.0008, 0.001, 0.001],[0.0003, 0.0004, 0.0006, 0.0007, 0.0007],[0.0003, 0.0006, 0.0006, 0.0008, 0.0009],[0.00005, 0.0001, 0.0002, 0.0002, 0.0003]]
e_test_rc = [[0.0355, 0.0519, 0.0536, 0.0579, 0.049],[0.038, 0.0533, 0.055, 0.0581, 0.0444],[0.0351, 0.038, 0.0343, 0.0346, 0.0367],[0.0262, 0.0268, 0.0211, 0.0236, 0.0225],[0.0273, 0.0284, 0.0273, 0.0286, 0.0272]]
data_e = [e_train_rmse,e_train_rc,e_test_rmse,e_test_rc]
names = ['train_rmse','train_rc','test_rmse','test_rc']
ylabels = ['RMSE (Mean)','Rank Correlation','RMSE (Mean)','Rank Correlation']
# Bars
for q in range(0,4):
    fig, ax = plt.subplots()
    for i in range (0,5):
        ax.bar(x+width*i,
               data[q][i],
               width,
               yerr=list(2*np.array(data_e[q][i])), align='center', ecolor='black', capsize=3, edgecolor='black')
    # Extras
    ax.set_ylabel(ylabels[q])
    ax.set_xticks(x+width*2)
    ax.set_xticklabels( ('Step-1','Step-2','Step-3','Step-4','Step-5') )
    ax.legend( ('FFNN', 'RNN', 'LSTM', 'Bayes-RNN', 'Bayes-LSTM') )
    plt.tight_layout(pad=1.0)
    plt.savefig(data_path + names[q] +'_bar_chart.png')
    # plt.show()

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