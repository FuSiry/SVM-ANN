import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
import numpy as np

def plotspc(x_col, data_x, tp):
    # figsize = 5, 3
    figsize = 8, 5.5
    figure, ax = plt.subplots(figsize=figsize)
    # ax = plt.figure(figsize=(5,3))
    x_col = x_col[::-1]  # 数组逆序
    y_col = np.transpose(data_x)
    plt.plot(x_col, y_col )
    plt.tick_params(labelsize=12)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    font = {'weight': 'normal',
            'size': 16,
            }
    plt.xlabel("Wavenumber/$\mathregular{cm^{-1}}$", font)
    plt.ylabel("Absorbance", font)
    # plt.title("The spectrum of the {} dataset".format(tp), fontweight="semibold", fontsize='x-large')
    plt.show()
    plt.tick_params(labelsize=23)

def TableDataLoad(tp, test_ratio, start, end, seed):

    # global data_x
    data_path = '..//Data//table.csv'
    Rawdata = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    table_random_state = seed

    if tp =='raw':
        data_x = Rawdata[0:, start:end]

        # x_col = np.linspace(0, 400, 400)
    if tp =='SG':
        SGdata_path = './/Code//TableSG.csv'
        data = np.loadtxt(open(SGdata_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        data_x = data[0:, start:end]
    if tp =='SNV':
        SNVata_path = './/Code//TableSNV.csv'
        data = np.loadtxt(open(SNVata_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        data_x = data[0:, start:end]
    if tp == 'MSC':
        MSCdata_path = './/Code//TableMSC.csv'
        data = np.loadtxt(open(MSCdata_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        data_x = data[0:, start:end]
    data_y = Rawdata[0:, -1]
    x_col = np.linspace(7400, 10507, 404)
    plotspc(x_col, data_x[:, :], tp=0)

    x_data = np.array(data_x)
    y_data = np.array(data_y)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_ratio,random_state=table_random_state)
    return X_train, X_test, y_train, y_test