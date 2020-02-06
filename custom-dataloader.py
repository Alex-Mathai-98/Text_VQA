import torchvision, torch, os
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
path = 'path/to/image/directory'
transform = torchvision.transforms.ToTensor()

#Import directory structure directly as a torch Tensor
def load_dataset(path):
    train_dataset = torchvision.datasets.ImageFolder(
        root=path,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False
    )
    return train_loader

#Dataloaders for each directory
dataset = load_dataset(path, transform)
