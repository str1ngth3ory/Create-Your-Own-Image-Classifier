import argparse

def get_predict_args():
    ''' Retrieve image file path, checkpoint data, training
        hyperparameters.
    '''
    parser = argparse.ArgumentParser(
        description='Retrieve user input and training hyperparameters...')
    
    parser.add_argument('image_dir', action='store',
                         help='Path to image, eg. ./image.jpg')
    parser.add_argument('checkpt_dir', action='store',
                         help='Directory of checkpoint, eg. ./')
    parser.add_argument('--top_k', action='store', default=5, type=int,
                         metavar='', help='Number of top classifications')
    parser.add_argument('--category_names', action='store', default='cat_to_name.json',
                         metavar='', help='File that stores the category names')
    parser.add_argument('--gpu', action='store_true',
                         help='Use GPU for training')
    
    return parser.parse_args()