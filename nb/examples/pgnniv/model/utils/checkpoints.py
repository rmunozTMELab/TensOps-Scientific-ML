import os
import torch

def save_checkpoint(model, optimizer, epoch, folder_path, end_flag=False, **kwargs):
    if end_flag:
        filename = os.path.join(folder_path, f'epoch_final.pth')
    else:
        filename = os.path.join(folder_path, f'epoch_{str(epoch)}.pth')
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_architecture': model.__dict__,  
    }
    
    for key, value in kwargs.items():
        state[key] = value

    torch.save(state, filename)


def load_checkpoint(model, optimizer, epoch, folder_path):

    filename = os.path.join(folder_path, f'epoch_{str(epoch)}.pth')
    checkpoint = torch.load(filename, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    kwargs = {key: checkpoint[key] for key in checkpoint if key not in ['model_state_dict', 'optimizer_state_dict', 'model_architecture']}
    
    return model, optimizer, kwargs

def load_results(model, optimizer, folder_path, map_location):

    filename = os.path.join(folder_path, f'epoch_final.pth')
    checkpoint = torch.load(filename, weights_only=False, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    kwargs = {key: checkpoint[key] for key in checkpoint if key not in ['model_state_dict', 'optimizer_state_dict', 'model_architecture']}
    
    return model, optimizer, kwargs