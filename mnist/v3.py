import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

train_file = "/kaggle/input/digit-recognizer/train.csv"
test_file = "/kaggle/input/digit-recognizer/test.csv"
sample_submission = "/kaggle/input/digit-recognizer/sample_submission.csv"

torch.cuda.is_available()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #x = x.view(-1, 1, 28, 28)

        x = self.relu1(self.conv1(x))
        x = F.pad(x, [2, 2, 2, 2])  # [left, right, top, bot]
        x = self.pool1(x)

        x = self.relu2(self.conv2(x))
        x = F.pad(x, [2, 2, 2, 2])  # [left, right, top, bot]
        x = self.pool2(x)

        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)


        return x
 
 class MnistDataset(Dataset):

    def __init__(self, x, y=None):
        self.data = x
        self.labels = y

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]
def read_train():
    data = pd.read_csv(train_file)

    train_y = data.values[:, 0]
    train_x = data.values[:, 1:].astype(np.float32)
    train_x = train_x.reshape([-1, 28, 28, 1]).transpose((0, 3, 1, 2))

    train_dataset = MnistDataset(train_x, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    return train_dataloader

def read_test():
    data = pd.read_csv(test_file)
    test_x = data.values[:, :].astype(np.float32)
    test_x = test_x.reshape([-1, 28, 28, 1]).transpose((0, 3, 1, 2))

    test_dataset = MnistDataset(test_x)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_dataloader

device = torch.device('cuda:0')
print(device)

def train():
    net = Net().to(device)

    criterion = nn.CrossEntropyLoss() # include softmax inside, not need extra softmax
   # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    print(net)

    dataloader = read_train()

    epoch = 1
    #iterations = 1

    for e in range(epoch):

        running_loss = 0.0
        total = 0
        correct = 0
        for step, (x_batch, y_batch) in enumerate(dataloader):

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = net(x_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 200 == 199:    # print every 100 mini-batches
                print('[epoch: %d, step: %5d] loss: %.3f' % (e + 1, step + 1, running_loss / 200))
                running_loss = 0.0

            _, predicted = torch.max(outputs.data, 1)
            #print(label_vals.__len__())
            total += y_batch.__len__()
            correct += (predicted == y_batch).sum().item()
            #print(correct)
            if step % 200 == 199:    # print every 100 mini-batches
                print(predicted)
                print(y_batch)
                print('[correct: %d, total: %5d], acc: %.3f' % (correct, total, correct / total))

    torch.save(net.state_dict(), '/kaggle/working/model.pth')

def val():
    batchsize = 1

    dataloader = read_test()

    net = Net().to(device)
    net.load_state_dict(torch.load('/kaggle/working/model.pth'))

    results = []
    with torch.no_grad():
        for step, (x_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device) 
            outputs = net(x_batch)
            outputs = outputs.to("cpu")
            results.extend(map(lambda x: np.argmax(x), outputs))

    print("Model Type is FP32")
    print("Test Batchsize is :{}".format(batchsize))
    #print("Average throughput is: {} images/second".format(iterations * batchsize * 1000 * 1000 / totalTime))
    with open("/kaggle/working/sample_submission.csv", 'w') as f:
        import csv
        spamwriter = csv.writer(f)
        spamwriter.writerow(['ImageId', 'Label'])
        for i in range(0, dataloader.__len__()):
            spamwriter.writerow([i + 1, results[i].numpy()])
train()
val()
