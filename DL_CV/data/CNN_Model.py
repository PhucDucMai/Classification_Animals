import torch
import torch.nn as nn
from dataset_animals import Dataset_Animal


class CNN(nn.Module):
    dataset = Dataset_Animal(root='./animals', train=True)

    def __init__(self, num_class=len(dataset.categories_animal)):
        super().__init__()
        self.conv1 = self.makeBlock(in_chanels=3, out_chanels=16)
        self.conv2 = self.makeBlock(in_chanels=16, out_chanels=32)
        self.conv3 = self.makeBlock(in_chanels=32, out_chanels=64)
        self.conv4 = self.makeBlock(in_chanels=64, out_chanels=64)
        self.conv5 = self.makeBlock(in_chanels=64, out_chanels=64)

        # fully connected layer: in_feature: kết quả cuối cùng sau khi flatten
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=3136, out_features=1024),
            nn.LeakyReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=num_class)
        )

    def makeBlock(self, in_chanels, out_chanels):
        return nn.Sequential(
            # conv2d phai chi ra 3 thu: in_chanels , out_channels, kernel_size
            nn.Conv2d(in_channels=in_chanels, out_channels=out_chanels, kernel_size=3, padding=1),

            # giữa conv và action nên có thêm 1 layer(batchnorm) để tránh overfitting
            nn.BatchNorm2d(num_features=out_chanels),

            # activation không thay đổi kích thước, chỉ thay đổi giá trị
            nn.LeakyReLU(),

            # conv2d phai chi ra 3 thu: in_chanels , out_channels, kernel_size
            nn.Conv2d(in_channels=out_chanels, out_channels=out_chanels, kernel_size=3, padding=1),

            # giữa conv và action nên có thêm 1 layer(batchnorm) để tránh overfitting
            nn.BatchNorm2d(num_features=out_chanels),

            # activation không thay đổi kích thước, chỉ thay đổi giá trị
            nn.LeakyReLU(),

            # pooling layer
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        batch_size, chanels, height, width = x.shape
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    modal = CNN()
    input = torch.rand(8, 3, 224, 224)
    output = modal.forward(input)
    print(output.shape)
