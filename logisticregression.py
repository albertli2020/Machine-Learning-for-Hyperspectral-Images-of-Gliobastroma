import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from statistics import median

control_training = []
control_testing = []
tumor_training = []
tumor_testing = []

ctrl_trn_images = [
    "March 26 Control C-H.ccdsfg",
    "April 8 Control N-H.ccdsfg",
]
tmr_trn_images = [
    "March 12 Tumor C-H.ccdsfg",
    "April 8 Tumor N-H.ccdsfg",
]
ctrl_tst_images = [
    "May 16 Control C-H.ccdsfg",
    "May 16 Control N-H.ccdsfg",
]
tmr_tst_images = [
    "May 16 Tumor C-H.ccdsfg",
    "May 16 Tumor N-H.ccdsfg",
]
for i in ctrl_trn_images:
    control_training += open(i, "r").readlines()
    # print(len(control_training))
for i in tmr_trn_images:
    tumor_training += open(i, "r").readlines()
    # print(len(tumor_training))
for i in ctrl_tst_images:
    control_testing += open(i, "r").readlines()
    # print(len(control_testing))
for i in tmr_tst_images:
    tumor_testing += open(i, "r").readlines()
    # print(len(tumor_testing))

# control_ch = open("May 16 Control C-H.ccdsfg", "r")
# control_nh = open("May 16 Control N-H.ccdsfg", "r")
# control = control_ch.readlines() + control_nh.readlines()
# tumor_ch = open("May 16 Tumor C-H.ccdsfg", "r")
# tumor_nh = open("May 16 Tumor N-H.ccdsfg", "r")
# tumor = tumor_ch.readlines() + tumor_nh.readlines()
# f = open("May 16 Control C-H.ccdsfg", "r")
# f = open("May 16 Control C-H.ccdsfg", "r")

def create_dataset(data):
    data = [list(map(float, l.split("\t")))[1:] for l in data]
    data = [np.array(l) for l in data if max(l) > median(l) + 100]
    return np.array(data)


control_training = create_dataset(control_training)
tumor_training = create_dataset(tumor_training)
control_testing = create_dataset(control_testing)
tumor_testing = create_dataset(tumor_testing)

print(control_training)
exit()

control = np.array(list(control_training) + list(control_testing))
tumor = np.array(list(tumor_training) + list(tumor_testing))

target = np.array(list(np.zeros(len(control))) + list(np.ones(len(tumor))))
data = np.array(list(control) + list(tumor))

# print(len(data), len(target))

# bc = datasets.load_breast_cancer()
# # print(bc)
# # x, y = bc.data, bc.target
x, y = data, target

n_samples, n_features = x.shape

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1234)
x_train = np.array(list(control_training) + list(tumor_training))
x_test = np.array(list(control_testing) + list(tumor_testing))
y_train = np.array(list(np.zeros(len(control_training))) + list(np.ones(len(tumor_training))))
y_test = np.array(list(np.zeros(len(control_testing))) + list(np.ones(len(tumor_testing))))

# scale
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# model
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

# loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(x_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(x_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
