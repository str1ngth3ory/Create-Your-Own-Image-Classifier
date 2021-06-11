import argparse

def get_train_args():
    ''' Retrieve user input and training hyperparameters.
    '''
    parser = argparse.ArgumentParser(
        description='Retrieve user input and training hyperparameters...')
    
    parser.add_argument('data_dir', action='store',
                         help='Directory of data, eg. flowers/')
    parser.add_argument('--save_dir', action='store', default='/',
                         metavar='', help='Save directory for checkpoints')
    parser.add_argument('--arch', action='store', default='vgg16',
                         metavar='', help='Select network architecture - VGG, Resnet, Alexnet')
    parser.add_argument('--learning_rate', action='store', default=0.001, type=float,
                         metavar='', help='Specify learning rate')
    parser.add_argument('--hidden_units', action='append', default=[], type=int,
                         metavar='', help='Add hidden layer units')
    parser.add_argument('--epochs', action='store', type=int,
                         metavar='', help='Specify epochs')
    parser.add_argument('--gpu', action='store_true',
                         help='Use GPU for training')
    parser.add_argument('--batch_size', action='store', default=32, type=int,
                         metavar='', help='Specify batch size for loading data')
    parser.add_argument('--p_dropout', action='store', default=0.2, type=float,
                         metavar='', help='Specify drop out probability')
    parser.add_argument('--valid_interval', action='store', default=100, type=int,
                         metavar='', help='Specify validation intevals')
    
    return parser.parse_args()