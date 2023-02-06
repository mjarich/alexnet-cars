import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from stanford_cars import StandfordCarsDataset

net = torch.jit.load('./model.pt')
net.eval()

class_names = pd.read_csv('./stanford-cars/class_names.csv')['class_names']

test_data = StandfordCarsDataset('./stanford-cars/cardatasettrain.csv', './stanford-cars/cars_train/')
testloader = DataLoader(test_data, batch_size=1, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for idx, (image, label) in enumerate(testloader):
    label = int(label)
    image_path = os.path.join(test_data.image_dir, test_data.images[idx])
    class_name = class_names[label]

    image = image.to(device)
    
    out = net(image)
    predicted_label = torch.argmax(out)
    predicted_class = class_names[int(predicted_label)]

    print(f'image: {image_path}')
    print(f'ground truth label: {label}')
    print(f'ground truth class: {class_name}')
    print(f'predicted label: {predicted_label}')
    print(f'predicted class: {predicted_class}')

    plt.imshow(image.cpu().numpy().reshape((227,227,3)))
    plt.show()

    



