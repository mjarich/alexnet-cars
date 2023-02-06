import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from stanford_cars import StandfordCarsDataset

net = torch.jit.load('./model.pt')

class_names = pd.read_csv('./stanford-cars/class_names.csv')

test_data = StandfordCarsDataset('./stanford-cars/cardatasettest.csv', './stanford-cars/cars_test/')
testloader = DataLoader(test_data, batch_size=4, shuffle=True)

for idx, image in enumerate(test_data):
    label = test_data.labels[idx]
    image_path = os.path.join(test_data.image_dir, test_data.images[idx])
    class_name = class_names[label]
    
    out = net(image)
    predicted_label = torch.argmax(out)
    predicted_class = class_names[predicted_label]

    print(f'image: {image_path}')
    print(f'ground truth label: {label}')
    print(f'ground truth class: {class_name}')
    print(f'net out: {out}')
    print(f'predicted label: {predicted_label}')
    print(f'predicted class: {predicted_class}')

    plt.imshow(image.numpy().reshape((227,227,3)))
    plt.show()

    



