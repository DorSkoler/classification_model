import os
import torch
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.style.use('ggplot')

def clean_data(df):
    """
    this functions removes those rows from the DataFrame for which there are
    no images in the dataset
    """
    drop_indices = []
    print('[INFO]: Checking if all images are present')
    for index, image_id in tqdm(df.iterrows()):
        if not os.path.exists(f"/images/{image_id.id}.jpg"):
            drop_indices.append(index)
    print(f"[INFO]: Dropping indices: {drop_indices}")
    df.drop(df.index[drop_indices], inplace=True)
    return df

# save the trained model to disk
def save_model(epochs, model, optimizer, criterion):
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, '../outputs/model.pth')
    
    # save the train and validation loss plots to disk
def save_loss_plot(train_loss, val_loss):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../outputs/loss.jpg')
    plt.show()
    
    
    