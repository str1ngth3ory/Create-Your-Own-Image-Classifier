import torch
from torchvision import datasets, transforms

def load_data(train_dir, valid_dir, test_dir, batch_size):
    
    # Define transformation for training and testing datasets
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, 
                                              batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_dataset, 
                                             batch_size=batch_size, shuffle=False)
    
    # Extract class to index
    cls_to_idx = train_dataset.class_to_idx
    
    return trainloader, validloader, testloader, cls_to_idx