import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torchvision.io import read_image

# https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
# https://github.com/BotechEngineering/StanfordCarsDatasetCSV for csv file containing labels (instead of matlab annotations)
class StandfordCarsDataset(Dataset):
    def __init__(self, annotations_file, image_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.image_dir = image_dir

        self.labels = self.annotations['Class'] - 1
        self.images = self.annotations['image']

        assert len(self.labels) == len(self.images)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        image = read_image(image_path)
        
        # dataset has some grayscale images
        if image.size()[0] == 1:
            image = torch.tile(image, (3, 1, 1))

        transform = Resize((227,227))
        image = transform(image).float() / 255
        return image, self.labels[index]
    

if __name__ == '__main__':
    d = StandfordCarsDataset('./stanford-cars/cardatasettrain.csv', './stanford-cars/cars_train/')
    print(len(d))
    print(d[1])