import json
import torch
from get_predict_args import get_predict_args
from load_checkpoint import load_checkpoint
from process_image import process_image

def get_key(class_dict, idx):
    ''' retrieve key based on value.
    '''
    for key, value in class_dict.items():
        if value == idx:
            return key
        
def predict(image, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.to(device)
    image = torch.unsqueeze(image, 0).to(device)
    model.eval()
    with torch.no_grad():
        log_ps = model(image)
        ps = torch.exp(log_ps)
        top_ps, top_idx = ps.topk(topk, dim=1)
    model.train()
    
    class_labels = []
    for class_idx in top_idx.reshape(-1).tolist():
        class_labels.append(get_key(model.class_to_idx, class_idx))
    
    return top_ps.reshape(-1).tolist(), class_labels

def main():
    
    # Retrieve image path, checkpoint file, and other user input
    predict_args = get_predict_args()
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() and predict_args.gpu else 'cpu')
    
    # Rebuild the model
    path_checkpt = predict_args.checkpt_dir + 'checkpoint.pth'
    model, epochs = load_checkpoint(path_checkpt)
    
    # Preprocess image
    image = process_image(predict_args.image_dir)
    
    # Load classification to numerical labels from json file
    with open(predict_args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # Predict the top k classifications and their probabilies
    top_ps, top_class_labels = predict(image, model, device, predict_args.top_k)
    
    print('Prediction of flower class:') 
    for class_label, class_p in zip(top_class_labels, top_ps):
        print(f'{cat_to_name[class_label]} - {class_p*100:.1f}%')
        
if __name__ == '__main__':
    main()