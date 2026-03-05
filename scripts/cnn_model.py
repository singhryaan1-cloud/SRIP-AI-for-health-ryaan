# %%
import torch
import pandas as pd
import csv
import numpy as np

# %%
path = "../Dataset/breathing_dataset.csv"
data = []
with open(path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)

# %%
data = data[2: ] #removing header and blank line

# %%
dataset_tuples = [(sample[0], sample[1:-1], sample[-1]) for sample in data]

# %%
X_np = [(tup[0], np.array(tup[1], dtype = np.float64).reshape((961,3))) for tup in dataset_tuples]
Y = [tup[2] for tup in dataset_tuples]


# %%
Y_unique = ['Hypopnea', 'Normal', 'Obstructive Apnea']
Y = [Y_unique.index(event) for event in Y]

# %%
class CNN1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(3,16,3, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(16,32,3,padding = 1),
            torch.nn.MaxPool1d(4),
            torch.nn.ReLU(),

            torch.nn.Conv1d(32,64,3, padding = 1),
            torch.nn.MaxPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),

            torch.nn.Linear(60*64,40),
            torch.nn.ReLU(),
            torch.nn.Linear(40,3)
        )
    def forward(self, x):
        return self.net(x)

# %%

#evaluation strategy
def create_training_test_set(X, Y, id):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i,sample in enumerate(X):
        if sample[0]!=id:
            X_train.append(sample[1])
            Y_train.append(Y[i])
        else:
            X_test.append(sample[1])
            Y_test.append(Y[i])
    X_test = np.array(X_test, dtype = np.float64)
    X_train = np.array(X_train, dtype = np.float64)
    Y_test = np.array(Y_test, dtype = np.int16)
    Y_train = np.array(Y_train, dtype = np.int16)
    
    p = np.random.permutation(len(X_train))
    return X_train[p], X_test, Y_train[p], Y_test

# %%
X_train, X_test, Y_train, Y_test = create_training_test_set(X_np, Y, "AP01")

# %%
len(Y_train[Y_train==2])

# %%
X_train = np.array(X_train)
X_train = np.transpose(X_train, [0,2,1])
X_test = np.array(X_test)
X_test = np.transpose(X_test, [0,2,1])

# %%
mean = [X_train[:,c,:].mean() for c in range(3)]
std = [X_train[:,c,:].std() for c in range(3)]
for c in range(3):
    X_train[:,c,:] = (X_train[:,c,:]-mean[c])/std[c]
for c in range(3):
    X_test[:,c,:] = (X_test[:,c,:]-mean[c])/std[c]

# %%
counts = np.array([np.sum(Y_train==c) for c in range(3)])
weights = 1.0 / counts       # inverse frequency
weights = weights / weights.sum() * 3
class_weights = torch.tensor(weights, dtype=torch.float32)
from torch.utils.data import TensorDataset, DataLoader
X_t = torch.tensor(X_train, dtype=torch.float32)
y_t = torch.tensor(Y_train, dtype=torch.int64)
data = TensorDataset(X_t, y_t)
loader = DataLoader(data, batch_size =32, shuffle = True)

model = CNN1D()
criterion = torch.nn.CrossEntropyLoss(weight = class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 40
best_loss = 100000
for i in range(num_epochs):
    total_loss = 0
    for xb,yb in loader:
        pred = model(xb)
        loss = criterion(pred,yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if total_loss<best_loss:
        best_cnn = model
    print(f"Epoch : {i+1}, Total Loss : {total_loss}")

# %%
X_tst = torch.Tensor(X_test)
Y_test = np.array(Y_test)
with torch.no_grad():
    logits = best_cnn(X_tst)
    preds = np.argmax(logits, axis = 1)
    acc = (preds==Y_test).float().mean()
print(f"Accuracy : {acc}")

# %%
from sklearn.metrics import classification_report
print(classification_report(Y_test, preds))


