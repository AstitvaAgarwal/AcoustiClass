import os   
import torch
from torch.utils.data import Dataset   
import pandas as pd    
import torchaudio
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
    ]

def predict (model, input , target , class_mapping):
    model.eval()  
    with torch.no_grad():          
        predictions = model(input)          
        predicted_index = predictions[0].argmax(0)       
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

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



def train_single_epoch(model , data_loader , loss_fn, optimiser, device):         

    for inputs , targets in data_loader:                                         

        inputs, targets = inputs.to(device) , targets.to(device)  
        
        predictions = model(inputs)
        loss = loss_fn(predictions , targets)    
        
        
       
        optimiser.zero_grad()             
        
        loss.backward()                   
        optimiser.step()                  
        
    print(f"loss:{loss.item()}")


def train(model , data_loader , loss_fn, optimiser, device ,epochs):       
    for i in range(epochs):
        print(f"Epochs {i+1}")
        train_single_epoch(model,data_loader,loss_fn ,optimiser , device)
        print("-------------------------------")
    print("training is done")



class UrbanSoundDataset(Dataset):    
    
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device): 
        self.annotations = pd.read_csv(annotations_file)   
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate= target_sample_rate
        self.num_samples= num_samples
        
    
    def __len__(self):         
        return len(self.annotations)
    
    def __getitem__(self, index):    
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)    
        
        signal = self._resample_if_necessary(signal,sr)
        signal = self._mix_down_if_necessary(signal)
      
        signal = self._cut_if_necessary(signal) 
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)         
        return signal, label
    
    def _cut_if_necessary(self, signal):
        
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal    
        
    def _right_pad_if_necessary(self,signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
       
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples) #(1,2)
            
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal   
    
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal        
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:    
            signal = torch.mean(signal,dim=0, keepdim = True)   
        return signal
    
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index,5]}"      
        path = os.path.join(self.audio_dir, fold ,self.annotations.iloc[index,0])  
        return path
    
    def _get_audio_sample_label (self, index):
        return self.annotations.iloc[index, 6] 
    
class CNNNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
       
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 3,
                stride = 1,
                padding = 2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels = 64,
                out_channels = 128,
                kernel_size = 3,
                stride = 1,
                padding = 2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
            )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128*5*4, 10)   
        self.softmax = nn.Softmax(dim = 1) 
    def forward(self, input_data):
        x= self.conv1(input_data)
        x= self.conv2(x)
        x= self.conv3(x)
        x= self.conv4(x)
        x= self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions
    
    
if __name__ == "__main__":
    ANNOTATIONS_FILE = r"C:\\Users\\DELL\\Desktop\\Astitva\\UrbanSound8K\\metadata\\UrbanSound8K.csv"
    AUDIO_DIR = r"C:\\Users\\DELL\\Desktop\\Astitva\\UrbanSound8K\\audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050 
    
    if torch.cuda.is_available():
        device = "cuda"
    else :
        device = "cpu"
    print(f"Using device: {device}")
    
    mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate= SAMPLE_RATE , n_fft=1024 , hop_length=512 , n_mels=64)
      
    
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE , NUM_SAMPLES , device)
    
    print(f"There are {len(usd)} samples in the dataset.")
    
    signal, label = usd[0]
        
    a = 1
    
    cnn = CNNNetwork()
    summary(cnn, (1,64,44))
    
if torch.cuda.is_available():
    device = "cuda"
else :
    device = "cpu"
print(f"Using {device} device")


mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate= SAMPLE_RATE , n_fft=1024 , hop_length=512 , n_mels=64)

usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE , NUM_SAMPLES , device)

train_data_loader = create_data_loader(usd, BATCH_SIZE )

cnn= CNNNetwork().to(device)
print(cnn)


loss_fn =nn.CrossEntropyLoss()                      
optimiser = torch.optim.Adam(cnn.parameters() , lr = LEARNING_RATE)              

train(cnn, train_data_loader, loss_fn, optimiser , device , EPOCHS)        

torch.save(cnn.state_dict(), "feedforwardnet.pth")                          
print("Trained feed forward net saved at feedforwardnet.pth") 

cnn = CNNNetwork()
state_dict = torch.load("feedforwardnet.pth")
cnn.load_state_dict(state_dict)


mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate= SAMPLE_RATE , n_fft=1024 , hop_length=512 , n_mels=64)
ANNOTATIONS_FILE_01 = r"C:\Users\DELL\Desktop\Astitva\TEST\OwnSound8K.csv"
usd_01 = UrbanSoundDataset(ANNOTATIONS_FILE_01, AUDIO_DIR, mel_spectogram, SAMPLE_RATE , NUM_SAMPLES , "cpu")

input , target = usd_01[0][0] , usd[0][1]   #[batch size , num channels , fr , time]

input.unsqueeze_(0)


predicted, expected = predict(cnn, input , target, class_mapping)


print(f"Predicted: '{predicted}', expected: '{expected}'")  