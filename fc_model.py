import torch
import torch.nn.functional as F
from torch import nn, optim

# A class for a full connected neural network
class Network(nn.Module):
    
    def __init__(self, n_input, n_output, hiddens, p_dropout=0.2):
        '''
        Initiate a full connected neural network with hidden layers.
        Arguments:
            n_input:    int, size of input layer
            n_output:   int, size of output layer
            hiddens:    list of integers, sizes of hidden layers
            p_dropout:  float, drop out probablity
        '''
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(n_input, hiddens[0])]) # Hidden Layer 1
        
        # More hidden layers
        hidden_sizes = zip(hiddens[:-1],hiddens[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in hidden_sizes])
        
        self.output = nn.Linear(hiddens[-1], n_output) # Output layer
        
        self.dropout = nn.Dropout(p=p_dropout) # Drop out function
        
    def forward(self, x):
        '''
        Runs forward pass on input, x. Returns network output.
        '''
        for each_hidden in self.hidden_layers: 
            x = F.relu(each_hidden(x))
            x = self.dropout(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)
    
# A function to set/replace classifier for the selected architecture
def set_classifier(model, n_features, n_hiddens, n_output, p_dropout, arch):
    '''
    Update the pretrained model with a full connected neural network classifer
    Arguments:
        model:      torchvision.models object, a pretrained neural network
        n_input:    int, size of input layer
        n_output:   int, size of output layer
        hiddens:    list of integers, sizes of hidden layers
        p_dropout:  float, drop out probablity
    '''
    # Lock down the parameters of pretrained model
    for param in model.parameters():
        param.requires_grad = False

    # Redefine the classifier with a two hidden layer full connected neural network
    classifier = Network(n_features, n_output, n_hiddens, p_dropout)

    # Replace the classifier in the pretrained neural network
    if arch == 'Resnet':
        model.fc = classifier
    else:
        model.classifier = classifier
    
# Define the Validation function
def validation(model, dataloader, criterion, device):
    '''
    Run validation feedforward pass on given dataset.
    Arguments:
        model:      torchvision.models object, a pretrained neural network
        dataloader: torchvision.datasets.DataLoader object, store validation/test dataset
        criterion:  loss function criterion
    '''    
    loss = 0
    accuracy = 0

    model.eval() # Disable dropout

    with torch.no_grad():

        for images, labels in dataloader:

            # Load data to specified device
            images, labels = images.to(device), labels.to(device) 

            # Forward pass + checking accuracy
            log_ps = model(images)
            loss += criterion(log_ps, labels)
            ps = torch.exp(log_ps)
            top_prob, top_class = ps.topk(1, dim=1)
            equal_results = (labels == top_class.view(labels.shape))

            accuracy += torch.mean(equal_results.type(torch.FloatTensor))
    
    model.train() # Toggle dropout back on
    
    return loss/len(dataloader), accuracy/len(dataloader)


# Define the training function
def train_model(model, optimizer, trainloader, validloader, criterion, epochs, valid_interval, device):
    '''
    Train neural network model with validation.
    Arguments:
        model:          torchvision.models object, a pretrained neural network
        trainloader:    torchvision.datasets.DataLoader object, store training dataset
        validloader:    torchvision.datasets.DataLoader object, store validation dataset
        criterion:      loss function criterion
        epochs:         int
        valid_interval: float
    '''    
       
    # Initialization
    train_losses, valid_losses = [], []
    step = 0
    running_loss = 0
    
    for epoch in range(epochs):
              
        for images, labels in trainloader:

            # Load data to specified device
            images, labels = images.to(device), labels.to(device) 

            step += 1
            optimizer.zero_grad() # Clear gradients from last cycle

            # Forward pass and backpropagation
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

#             # Monitoring the steps
#             if step % 20 == 0:
#                 print(f'Running step {step}')
                
            # Validate network at specified step intervals
            if step % valid_interval == 0:

                valid_loss, valid_accuracy = validation(model, validloader, criterion, device)

                # Store losses for monitoring overfitting
                train_losses.append(running_loss/valid_interval)
                valid_losses.append(valid_loss)

                # Print out training and validation results at specified step intervals
                print(f'Epoch {epoch+1:2d} of {epochs:2d} Step {step:3d} ...',
                      f'Train Loss: {running_loss/valid_interval:.3f} ..',
                      f'Validation Loss: {valid_loss:.3f} ..',
                      f'Accuracy: {valid_accuracy:.3f}')

                running_loss = 0
    
    return train_losses, valid_losses