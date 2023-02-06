import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, name: str, num_classes: int) -> None:
        super().__init__()
        self.name = name
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.maxpool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.maxpool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)

        self.maxpool3 = nn.MaxPool2d(3, 2)

        self.dense1 = nn.Linear(6*6*256, 4096)
        self.dense2 = nn.Linear(4096, 4096)
        self.dense3 = nn.Linear(4096, self.num_classes)
       
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool3(x)
        x = torch.reshape(x, (-1, 6*6*256))
        x = self.dropout(F.relu(self.dense1(x)))
        x = self.dropout(F.relu(self.dense2(x)))
        x = self.dense3(x)
        return x

    def save(self):
        model_scripted = torch.jit.script(self)
        model_scripted.save('./model.pt')


if __name__ == "__main__":
    from train import train
    from torch.utils.data import DataLoader
    from stanford_cars import StandfordCarsDataset
    
    print('building AlexNet...')
    net = AlexNet('test', 196)

    print('getting test output...')
    print(f'test output: {net(torch.randn((10, 3, 227, 227)))}')

    print('generating training dataset...')
    training_data = StandfordCarsDataset('./stanford-cars/cardatasettrain.csv', 
                                         './stanford-cars/cars_train/')
    trainloader = DataLoader(training_data, batch_size=128, shuffle=True)

    print('beginning training...')
    train(net, trainloader, learn_rate=1e-4, num_epochs=40, report_freq=10)
    
    print('saving model...')
    net.save()

