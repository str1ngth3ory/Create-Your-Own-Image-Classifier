import torch
from torchvision import models
from fc_model import Network

def load_checkpoint(file):
    
    # Read checkpoint file
    checkpoint = torch.load(file, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Load the user specified architecture
    if checkpoint['arch'] == 'Resnet':
        model = models.resnet18(pretrained=True)
    elif checkpoint['arch'] == 'Alexnet':
        model = models.alexnet(pretrained=True)
    elif checkpoint['arch'] == 'VGG':
        model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    # Rebuild the fc model   
    classifier = Network(checkpoint['n_input'],
                         checkpoint['n_output'],
                         checkpoint['n_hiddens'],
                         checkpoint['p_dropout'])
    

    classifier.load_state_dict(checkpoint['state_dict'])

    # replace the classifier
    if checkpoint['arch'] == 'Resnet':
        model.fc = classifier
    else:
        model.classifier = classifier
    
    model.class_to_idx = checkpoint['class_to_idx']
    
#     optimizer.load_state_dict(checkpoint['optim_state'])   # Not using this currently
    
    epochs = checkpoint['epochs']
    
    return model, epochs