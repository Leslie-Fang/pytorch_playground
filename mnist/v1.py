import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim

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

        x = self.relu1(self.conv1(x))
        x= F.pad(x, [2, 2, 2, 2])  # [left, right, top, bot]
        x = self.pool1(x)

        x = self.relu2(self.conv2(x))
        x = F.pad(x, [2, 2, 2, 2])  # [left, right, top, bot]
        x = self.pool2(x)

        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def read_train():
    data = pd.read_csv("./dataset/train.csv")
    data = data.values #pandas dataframe 转换成 numpy.ndarray 格式
    data[:, 1:785] = data[:, 1:785].astype(np.float32)/128.0-1
    return data

def train():
    net = Net()

    criterion = nn.CrossEntropyLoss() # include softmax inside, not need extra softmax
   # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    print(net)

    raw_data = read_train()
    print(raw_data.shape)

    batchsize = 64
    epoch = 1
    iterations = int(raw_data.shape[0] / batchsize)
    #iterations = 1

    for e in range(epoch):
        np.random.shuffle(raw_data)
        train_label = raw_data[:, 0].astype(np.int32)
        train_data = raw_data[:, 1:785]

        print("iterations in one epoch:{} is: {}".format(e, iterations))

        running_loss = 0.0
        total = 0
        correct = 0
        for step in range(iterations):

            # data preprocess
            start = step * batchsize
            end = (step + 1) * batchsize

            train_image_pixs = train_data[start:end, :]
            label_vals = train_label[start:end]

            train_image_pixs = np.array(train_image_pixs, dtype=np.float32).reshape([-1, 28, 28, 1]).transpose((0,3,1,2))
            train_x = torch.FloatTensor(train_image_pixs)
            train_y = torch.LongTensor(np.array(label_vals, dtype=np.int))

            optimizer.zero_grad()
            outputs = net(train_x)

            loss = criterion(outputs, train_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 200 == 199:    # print every 100 mini-batches
                print('[epoch: %d, step: %5d] loss: %.3f' % (e + 1, step + 1, running_loss / 200))
                running_loss = 0.0

            _, predicted = torch.max(outputs.data, 1)
            #print(label_vals.__len__())
            total += label_vals.__len__()
            correct += (predicted == train_y).sum().item()
            #print(correct)
            if step % 200 == 199:    # print every 100 mini-batches
                print(predicted)
                print(train_y)
                print('[correct: %d, total: %5d], acc: %.3f' % (correct, total, correct / total))

    torch.save(net.state_dict(), './save/model.pth')

def read_test():
    data = pd.read_csv("./dataset/test.csv")
    data = data.values/128.0-1
    print(data.shape)
    return data

def val():
    batchsize = 1

    raw_data = read_test()
    input_image_size = raw_data.shape[1]

    net = Net()
    net.load_state_dict(torch.load('./save/model.pth'))

    iterations = int(raw_data.shape[0] / batchsize)
    results = []
    with torch.no_grad():
        for step in range(iterations):
            start = step * batchsize
            end = (step + 1) * batchsize
            inference_image_pixs = raw_data[start:end, :]
            inference_image_pixs = np.array(inference_image_pixs, dtype=np.float32).reshape([-1, 28, 28, 1]).transpose((0,3,1,2))
            inference_x = torch.FloatTensor(inference_image_pixs)
            outputs = net(inference_x)
            results.extend(map(lambda x: np.argmax(x), outputs))

    print("Model Type is FP32")
    print("Test Batchsize is :{}".format(batchsize))
    #print("Average throughput is: {} images/second".format(iterations * batchsize * 1000 * 1000 / totalTime))
    with open("./sample_submission.csv", 'w') as f:
        import csv
        spamwriter = csv.writer(f)
        spamwriter.writerow(['ImageId', 'Label'])
        for i in range(0, iterations * batchsize):
            spamwriter.writerow([i + 1, results[i].numpy()])



if __name__ == "__main__":
    train()
    val()
