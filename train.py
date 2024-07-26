import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
import torch.optim as op
import pandas as pd
import os
import tempfile

class Bottleneck(nn.Module):
    """
    Follows Bottleneck architecture for a Residual Block:
    1 Conv layer with 1x1 kernel, padding of 0 (changes output channels to out_channels)
    1 Conv layer with 3x3 kernel, padding of 1 (maintains channels and convolves)
    1 Conv layer with 1x1 kernel, padding of 0 (changes output channels to expansion*out_channels)
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsampling: nn.Module = None) -> None:
        """
        PARAMETERS
        --------------------------------
        in_channels: expected number of channels of the input tensor

        out_channels: output number of channels after the first conv layer

        stride: stride parameter inputed into the second conv layer

        downsampling: a nn.Module that corrects the spatial dimensions and depth of the identity if
        there is a difference between it and the output tensor 
        """
        super().__init__() # defines this class as a nn.Module (can input and output tensors)
        self.relu = nn.ReLU() # defines a ReLU layer (nn.Module as well)
        self.downsampling = downsampling # assigns a downsampling if given

        # [1, 1, 6, 6]

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_1 = nn.BatchNorm2d(out_channels)
        # defines first convolutional layer

        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm_2 = nn.BatchNorm2d(out_channels)
        # defines second convolutional layer

        self.conv_3 = nn.Conv2d(out_channels, self.expansion*out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_3 = nn.BatchNorm2d(self.expansion*out_channels)
        # defines third convolutional layer
    

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of a Bottleneck block and is called when a tensor is inputed
        as the sole argument to a Bottleneck object
        """
        identity = tensor.clone()
        # to perform identity addition later on

        tensor = self.relu(self.norm_1(self.conv_1(tensor)))

        tensor = self.relu(self.norm_2(self.conv_2(tensor)))

        tensor = self.norm_3(self.conv_3(tensor))
        if self.downsampling:
            identity = self.downsampling(identity)
        # perform downsampling on identity so both tensor and identity have the same shape

        tensor += identity
        # perform identity addition

        tensor = self.relu(tensor)

        return tensor


class ResNet(nn.Module):
    """
    Follows ResNet architecture depending on block formation:
    Initial config layer is structured as:
    1 Conv layer with 7x7 kernel, stride of 2, padding = 3, output channels = 64
    """
    def __init__(self, Block, block_structure, num_classes, input_channels=3) -> None:
        """
        PARAMETERS
        --------------------------------
        Block: a Residual Block structure, like the Bottleneck class

        block_structure: gives the number of blocks within each of the 4 layers, for ResNet50 this is [3, 4, 6, 3]

        num_classes: an integer for the expected number of classes of the input dataset

        input_channels (optional): the number of channels of the input dataset, set to 3 by default
        """
        super().__init__()
        self.conv_1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm_1 = nn.BatchNorm2d(64)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.cur_channels = 64
        # config layer

        self.layer_1 = self.create_layer(Block, block_structure[0], out_channels=64, stride=1)
        self.layer_2 = self.create_layer(Block, block_structure[1], out_channels=128, stride=2)
        self.layer_3 = self.create_layer(Block, block_structure[2], out_channels=256, stride=2)
        self.layer_4 = self.create_layer(Block, block_structure[3], out_channels=512, stride=2)
        # 4 layer modules

        self.adapt_pool = nn.AdaptiveAvgPool2d(1)
        self.fully_connected = nn.Linear(in_features=512*Block.expansion, out_features=num_classes)
        # adapt_pool converts our tensor into a vector, and fully_connected uses this vector to compute a score
        # for each class number 
    

    def forward(self, tensor):
        """
        Defines the forward pass for a ResNet model. Called when a tensor is passed as the sole input
        into a instantized ResNet model.

        Input tensor: (batch number, channels, height, width)
        """
        tensor = tensor.float()
        tensor = self.relu(self.norm_1(self.conv_1(tensor)))
        tensor = self.max_pooling(tensor)
        # initial configuration

        tensor = self.layer_1(tensor)
        tensor = self.layer_2(tensor)
        tensor = self.layer_3(tensor)
        tensor = self.layer_4(tensor)
        # forward pass through the 4 module layers

        tensor = self.adapt_pool(tensor)
        tensor = tensor.reshape(tensor.shape[0], -1)
        tensor = self.fully_connected(tensor)
        # reshapes tensor and feeds into fully-connected layer

        return tensor
    

    def create_layer(self, Block, num_blocks, out_channels, stride=1):
        potential_downsample = None
        all_blocks = []

        if stride != 1 or self.cur_channels != out_channels*Block.expansion:
            potential_downsample = nn.Sequential(
                    nn.Conv2d(self.cur_channels, out_channels*Block.expansion, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels*Block.expansion)
                )
        # create downsample for identity if needed

        all_blocks.append(Block(self.cur_channels, out_channels, stride=stride, downsampling=potential_downsample))
        self.cur_channels = out_channels*Block.expansion
        # reset cur_channels so that there will be no need for further downsampling in this layer (output of first
        # block will be fed into second block)

        for _ in range(num_blocks-1):
            all_blocks.append(Block(self.cur_channels, out_channels, stride=1, downsampling=None))
        layer = nn.Sequential(*all_blocks)
        # add rest of the blocks, and put in a sequence layer

        return layer


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.labels = pd.read_csv(os.path.join(img_dir, "classes.csv"))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.labels.iloc[index, 0])
        ground_truth = self.labels.iloc[index, 1]
        image = io.read_image(img_path)
        if self.transform:
            image = self.transform(image)
        
        return image, ground_truth


def train(model, epochs, dataloaders, dataset_sizes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently using {device} to run training")
    print()
    # detect gpu presence, default is cpu
    
    loss_function = nn.CrossEntropyLoss() # can implement manual weighing later
    optimizer = op.Adam(model.parameters(), lr=0.00001)
    model.to(device)

    with tempfile.TemporaryDirectory() as temp_dir:
        best_model_path = os.path.join(temp_dir, "best_model.pt")
        torch.save(model.state_dict(), best_model_path)
        best_acc = 0.0

        for epoch_num in range(epochs):
            print(f"Epoch {epoch_num} / {epochs-1}")
            print("------------------------------------------")

            for phase in ["train", "validate"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                
                total_loss = 0.0
                total_corrects = 0
                count = 0

                for image_tensor, ground_truth in dataloaders[phase]:
                    # count += 1
                    # if count % 30 == 0:
                    #     print(f"phase: {phase}, #{count}")
                    image_tensor, ground_truth = image_tensor.to(device), ground_truth.to(device)
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == "train"):
                        logits_output = model(image_tensor)
                        avg_loss = loss_function(logits_output, target=ground_truth)
                        _, prediction_tensor = torch.max(logits_output, 1)
                        # if phase == "train":
                        #     print(f"phase: {phase}, tensor: {logits_output}")

                        if phase == "train":
                            avg_loss.backward()
                            optimizer.step()
                    
                    total_loss += avg_loss.item() * image_tensor.size(0)
            
                    total_corrects += torch.sum(prediction_tensor == ground_truth)

                avg_phase_loss = total_loss / dataset_sizes[phase]
                avg_phase_acc = total_corrects.to(torch.float64) / dataset_sizes[phase]
                print(total_corrects, dataset_sizes[phase])
                print(f"{phase} loss: {avg_phase_loss},    {phase} accuracy: {avg_phase_acc}")

                if phase == "validate" and avg_phase_acc > best_acc:
                    best_acc = avg_phase_acc
                    torch.save(model.state_dict(), best_model_path)

        print()
        print(f"Best validation accuracy: {best_acc}")
        model.load_state_dict(torch.load(best_model_path))
    
    return model
            

            

if __name__ == "__main__":
    # ten = torch.rand((3, 3, 6, 6))
    # # x = Bottleneck(1, 2)
    # y = ResNet(Bottleneck, [1,1,1,1], 3)
    # for param in y.parameters():
    #     print(param)

    val = ImageDataset("./validate")
    trains = ImageDataset("./train")
    dataval = DataLoader(val, batch_size=32)
    datatrain = DataLoader(trains, batch_size=32)
    dataloaders = {"train": datatrain,
                   "validate": dataval}
    dataset_sizes = {"train": len(trains),
                     "validate": len(val)}
    resnet_model = ResNet(Bottleneck, [3,4,6,3], 2)
    print("setup done")
    model = train(resnet_model, 15, dataloaders=dataloaders, dataset_sizes=dataset_sizes)
    torch.save(model.state_dict(), "./test.pt")
    print("done!")
    
    # res = y(ten)
    # softmax = nnf.softmax(res, dim=1)
    # print(softmax)
    pass