import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from stanford_cars import StandfordCarsDataset

net = torch.jit.load('./model.pt')
net.eval()

class_names = pd.read_csv('./stanford-cars/class_names.csv')['class_names']

test_data = StandfordCarsDataset('./stanford-cars/cardatasettest.csv', './stanford-cars/cars_test/')
testloader = DataLoader(test_data, batch_size=1, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_correct = 0

for idx, (image, label) in enumerate(testloader):
    label = int(label)
    image_path = os.path.join(test_data.image_dir, test_data.images[idx])
    class_name = class_names[label]

    image = image.to(device)
    
    out = net(image)
    predicted_label = int(torch.argmax(out))
    predicted_class = class_names[predicted_label]

    print(f'image: {image_path}')
    print(f'ground truth label: {label}')
    print(f'ground truth class: {class_name}')
    print(f'predicted label: {predicted_label}')
    print(f'predicted class: {predicted_class}')
    print('*' * 50)

    if label == predicted_label:
        num_correct += 1

print(f'accuracy: {num_correct / len(test_data):.3f}')

