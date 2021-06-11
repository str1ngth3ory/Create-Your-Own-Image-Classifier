import matplotlib.pyplot as plt
import json
import torch
from torch import nn, optim
from torchvision import models
from get_train_args import get_train_args
from load_data import load_data
from fc_model import Network, set_classifier, validation, train_model
from workspace_utils import keep_awake # To keep session active


def main():
    # Retrieve user input and training hyperparameters
    train_args = get_train_args()
    arch = train_args.arch
    
    # Define directories for data
    data_dir = train_args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Load data
    trainloader, validloader, testloader, cls_to_idx = load_data(train_dir, valid_dir, test_dir, train_args.batch_size)

    # Load class to name
    with open('cat_to_name.json') as f:
        cat_to_name = json.load(f)
    
    # Check if CUDA is available and whether to run model on it   
    device = torch.device('cuda' if torch.cuda.is_available() and train_args.gpu else 'cpu')
    
    # Load the user specified architecture
    if arch == 'Resnet':
        model = models.resnet18(pretrained=True)
    elif arch == 'Alexnet':
        model = models.alexnet(pretrained=True)
    elif arch == 'VGG':
        model = models.vgg16(pretrained=True)

    # Define hyperparameters of the neural network
    # Get the number of features from architecture
    if arch == 'Resnet':
        n_features = model.fc.in_features
    elif arch == 'Alexnet':
        n_features = model.classifier[1].in_features
    elif arch == 'VGG':
        n_features = model.classifier[0].in_features
    n_hiddens = train_args.hidden_units
    n_output = len(cat_to_name) # number of flower categories
    p_dropout = train_args.p_dropout
    
    # Set/update classifier in the model
    set_classifier(model, n_features, n_hiddens, n_output, p_dropout, arch)

    # Define the loss criterion function
    criterion = nn.NLLLoss()

    model.to(device) # Load model to specified device

    for i in keep_awake(range(1)):

        # Assign training hyperparameters
        learn_rate = train_args.learning_rate
        epochs = train_args.epochs
        valid_interval = train_args.valid_interval

        # Define optimizer and assign/update learn rate
        if arch == 'Resnet':
            optimizer = optim.Adam(model.fc.parameters(), lr=learn_rate)
        else:
            optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

        # Train network (with validation)
        train_losses, valid_losses = train_model(model, optimizer, trainloader, validloader, criterion, epochs, valid_interval, device)

#     # Plot training and validation history
#     plt.plot(train_losses, label='Training loss')
#     plt.plot(valid_losses, label='Validation loss')
#     plt.legend(frameon=False)

    model.class_to_idx = cls_to_idx # Attach class to index information to the model for future inference
    
    # Extract checkpoint information
    checkpoint = {'n_input': n_features,
                  'n_output': n_output,
                  'n_hiddens': n_hiddens,
                  'p_dropout': p_dropout,
                  'epochs': epochs,
                  'arch': train_args.arch,
                  'class_to_idx': model.class_to_idx,
              # 'optim_state': optimizer.state_dict(), # Keeping it off to save disk space
                  'state_dict': model.fc.state_dict() if arch == 'Resnet' else model.classifier.state_dict()}

    # Save checkpoint
    torch.save(checkpoint, 'checkpoint.pth')
    
    
# Call main function
if __name__ == '__main__':
    main()