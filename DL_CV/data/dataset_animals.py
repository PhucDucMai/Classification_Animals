import os
from torch.utils.data import DataLoader, Dataset
import cv2
from torchvision.transforms import Compose, ToTensor, Resize
import warnings
warnings.filterwarnings('ignore')


class Dataset_Animal(Dataset):
    def __init__(self, root, train, transform=None):
        # get the path to the image folder train or test
        if train:
            self.path_to_image = os.path.join(root, "train")
        else:
            self.path_to_image = os.path.join(root, "test")

        # This list to save the list of animals in the folder
        self.categories_animal = []
        for animal in os.listdir(self.path_to_image):
            self.categories_animal.append(animal)
        # print(self.categories_animal)

        # This list to save paths to the image and labels of images
        self.images_file = []
        self.lables = []
        for category in self.categories_animal:
            image_directory = os.path.join(self.path_to_image, category)
            for image_name in os.listdir(image_directory):
                path_to_the_image = os.path.join(image_directory, image_name)
                self.images_file.append(path_to_the_image)
                self.lables.append(self.categories_animal.index(category))

        # print(len(self.images_file))
        # print(len(self.lables))

        self.transform = transform

    # get the length
    def __len__(self):
        return len(self.lables)

    # get the image and image's label
    def __getitem__(self, index):
        image = cv2.imread(self.images_file[index])
        if self.transform:
            image = self.transform(image)
        label = self.lables[index]
        return image, label


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])

    dataset = Dataset_Animal(root='./animals', train=True, transform=transform)
    dataloader = DataLoader(dataset=dataset, shuffle=True, drop_last=True, batch_size=8, num_workers=6)
    # image, label = dataset.__getitem__(12000)
    for images, labels in dataloader:
        print(images.shape)
        print(labels)
