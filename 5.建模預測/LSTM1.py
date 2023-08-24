import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import csv
from torch.utils.data import Dataset, DataLoader

# 刪除欄位: '日期','oven_id','layer_id'
pd.set_option('display.max_columns', None)
historydf = pd.read_csv('data.csv')
OUTPUT_LINE_TITLE = np.array(historydf)[:190, 1:3]
OUTPUT_LINE_TITLE = ['{}-{}'.format(line[0], line[1]) for line in OUTPUT_LINE_TITLE]
historydf.dropna(how='any', inplace=True)
historydf = historydf.drop(['日期', 'oven_id', 'layer_id'], axis=1)
historydf = historydf.astype(float)

NUM_OF_DATE = 28

# 切分資料


def data_split(data, sample):
    # 資料維度: 壞掉燈管數(低功率),壞掉燈管數(高功率)
    # 進水量,進水溫,出水溫,水冷板A溫度,水冷板B溫度,累積時數,燈管功率(低功率),燈管功率(高功率)... 10維
    feature = len(data.columns)

    # 將dataframe 轉成 numpy array
    data = data.values
    newdata = []
    y_data = []

    # 取第0-19天(訓練)+第20天的壞掉燈管數(預測目標)、1-20+21...等
    # range(len(data)-sample)
    for i in range(73):  # 1~28  2~29 ... 74~101   test 102~129
        # i為每一天
        newdata.append(data[i * 190:(i * 190 + sample)])
        y_data.append(data[i * 190 + sample:i * 190 + sample + sample, 0:2])
       # y_data.append(data[i+sample,0:2])

    # 取 result 的前 80% instance做為訓練資料
    #n_train = round(0.8*len(newdata))

    newdata = np.array(newdata)
    x_train = newdata  # [:int(n_train)]     # 74 ~ 101
    y_train = y_data  # [:int(n_train)]      # 102 ~ 129
    #x_test = newdata[int(n_train):]
    #y_test = y_data[int(n_train):]

    x_test = data[-NUM_OF_DATE * 2 * 190:-NUM_OF_DATE * 190]
    y_test = data[-NUM_OF_DATE * 190:, 0:2]

    return x_train, np.array(y_train), x_test, np.array(y_test)

class dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_sample = len(x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_sample


# 以NUM_OF_DATE天為一區間進行預測
x_train, y_train, x_test, y_test = data_split(historydf, NUM_OF_DATE * 190)

x_train = torch.from_numpy(x_train.reshape(-1, NUM_OF_DATE, 1900)).type(torch.Tensor)
y_train = torch.from_numpy(y_train.reshape(-1, NUM_OF_DATE, 380)).type(torch.Tensor)

x_test = torch.from_numpy(x_test.reshape(-1, NUM_OF_DATE, 1900)).type(torch.Tensor)
y_test = torch.from_numpy(y_test.reshape(-1, NUM_OF_DATE, 380)).type(torch.Tensor)


RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.rnn = nn.LSTM(input_size=input_dim,
                           hidden_size=hidden_dim,
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=False)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化
        output, (hn, cn) = self.rnn(x, None)
        out = self.out(output)
        return out


if __name__ == "__main__":
    input_dim = 1900
    hidden_dim = 100
    num_layers = 2
    output_dim = 380
    num_epochs = 100
    learning_rate = 0.01
    batch = 10

    train_set = dataset(x_train, y_train)
    train_dataloader = DataLoader(dataset=train_set, batch_size=batch, shuffle=False)

    my_model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    my_model = my_model.to(DEVICE)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()

    # 迭代100次
    # range(num_epochs)
    for e in range(num_epochs):
        for i, data in enumerate(train_dataloader, 0):  # 迴圈取出批次資料，從0開始
            x_train, y_train = data
            x_train, y_train = x_train.to(DEVICE), y_train.to(DEVICE)
        y_train_pred = my_model(x_train)

        loss = loss_func(y_train_pred, y_train)

        # 將梯度歸零
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # 更新參數
        optimizer.step()
        print('Epoch: {}, Loss:{:.5f}'.format(e + 1, loss.item()))


    x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)
    y_test_pred = my_model(x_test)

    loss = loss_func(y_test_pred, y_test)

    print('loss = ', loss.item())
    y_test_pred = y_test_pred.data.cpu()
    output_data = y_test_pred.detach().numpy()

    y_test = y_test.data.cpu()
    y_test = y_test.detach().numpy()

    all_data = []

    c = np.zeros((1, 380))
    for y in range(len(output_data[-1])):
        c += output_data[-1, y]
    #c =  np.around(c)
    c = np.ceil(c)
    d = np.zeros((1, 380))
    for y in range(len(y_test[-1])):
        d += y_test[-1, y]

    last = []
    ans = []
    for i in range(190):
        ans.append(d[0, i * 2] + d[0, i * 2 + 1])
        last.append(c[0, i * 2] + c[0, i * 2 + 1])

    print(ans) # 標準答案
    print(last) #我們的輸出
    print((np.sum((np.array(last) - np.array(ans)) ** 2) / 190)**0.5)
    with open('answer.csv', 'w', newline='') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(['id', 'anomaly_total_number'])
        for i in range(len(OUTPUT_LINE_TITLE)):
            csvwriter.writerow([OUTPUT_LINE_TITLE[i], last[i]])
        