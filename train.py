import torch
from torch import nn
from torch.utils.data import DataLoader
from Creating_a_custom_dataset import UrbanSoundDataset
import torchaudio
from cnn import CNNNetwork

BATCH_SIZE = 128    
EPOCHS =10
LEARNING_RATE = .001

ANNOTATIONS_FILE = r"C:\\Users\\DELL\\Desktop\\Astitva\\UrbanSound8K\\metadata\\UrbanSound8K.csv"
AUDIO_DIR = r"C:\\Users\\DELL\\Desktop\\Astitva\\UrbanSound8K\\audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050 

def create_data_loader(train_data,batch_size):
    train_dataloader = DataLoader(train_data, batch_size = batch_size)
    return train_dataloader


#training the model :--
def train_single_epoch(model , data_loader , loss_fn, optimiser, device):          #training one epoch of your model
#loop through all the samples in the dataset and in each iterations will get a new batch of samples
    for inputs , targets in data_loader:                                         

        inputs, targets = inputs.to(device) , targets.to(device)  #assigning
        
        predictions = model(inputs)
        loss = loss_fn(predictions , targets)    #calculating loss  --> will compare these two and come up with the loss
        
        
        #at every iteration the optimiser is gonna claculate a gradience that will need to decide the updation of weights , so the gradience at each iteration gets saved
        optimiser.zero_grad()             #at each training iteration resets the gradience to zero so to start from scratch
        
        loss.backward()                   #backpropogating loss
        optimiser.step()                  #update weights
        
    print(f"loss:{loss.item()}")


def train(model , data_loader , loss_fn, optimiser, device ,epochs):       #this fnc will go through all the epochs that we want to train the model for 
    for i in range(epochs):
        print(f"Epochs {i+1}")
        train_single_epoch(model,data_loader,loss_fn ,optimiser , device)
        print("-------------------------------")
    print("training is done")

#End of training




if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else :
        device = "cpu"
    print(f"Using {device} device")
    
    #instantiating our dataset object and create data loader 
    mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate= SAMPLE_RATE , n_fft=1024 , hop_length=512 , n_mels=64)
    
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE , NUM_SAMPLES , device)

    train_data_loader = create_data_loader(usd, batch_size=BATCH_SIZE )
    
    cnn= CNNNetwork().to(device)
    print(cnn)
    
    
    
    
    loss_fn =nn.CrossEntropyLoss()                      
    optimiser = torch.optim.Adam(cnn.parameters() , lr = LEARNING_RATE)              
    
    train(cnn, train_data_loader, loss_fn, optimiser , device , EPOCHS)         #train model
    
    
    #storing after training  ||   state_dict() --> python dictionary that has all imp info about layers and parameters that have been trained 
    torch.save(cnn.state_dict(), "feedforwardnet.pth")                          
    print("Trained feed forward net saved at feedforwardnet.pth") 
    
    
    