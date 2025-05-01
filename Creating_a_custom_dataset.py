import os   #reading, writing files, creating, deleting directories,getting the current working directory.
import torch
from torch.utils.data import Dataset   #importing dataset --> this dataset is the base class we need for custom dataset
import pandas as pd    #used to load csv file 
import torchaudio


class UrbanSoundDataset(Dataset):    #this class will inherit from Dataset
    
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device): #constructor #audio_dir --> path where we store all the audio samples 
        self.annotations = pd.read_csv(annotations_file)   #reading csv file with help of pandas
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate= target_sample_rate
        self.num_samples= num_samples
        
    
    def __len__(self):          #check length of dataset || len(a) --> gives the length of 'a' 
        return len(self.annotations)
    
    def __getitem__(self, index):     #to obtain items from dataset  || a_list[1] == a_list.__getitem__(1)  --> both are same 
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)    #loading the audio file
        signal = signal.to(self.device)   #using CPU or GPU
        #signal -> (num_channels, samples)-> (2, 10000) -> (1,10000)
        signal = self._resample_if_necessary(signal,sr) #resample the data so that all will have same sample rate 
        signal = self._mix_down_if_necessary(signal)# convert the signal to mono
        #the number of samples should be equal to num_samples
        signal = self._cut_if_necessary(signal) #if num of samples are more than expected
        signal = self._right_pad_if_necessary(signal)# the num of samples are less than expected then we apply right padding 
        signal = self.transformation(signal)          #passing the signal to transform into mel spectrogram --> mel_spectrogram(signal)
        return signal, label
    
    def _cut_if_necessary(self, signal):
        #signal -> tensor-> (1, num_samples)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal    
        
    def _right_pad_if_necessary(self,signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
        #[1,1,1] -> [1,1,1,0,0]  = right padding
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples) #(1,2)
            #[1,1,1] -> [0,1,1,1,0,0]
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal   
    
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal        
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:    #(2,10000)
            signal = torch.mean(signal,dim=0, keepdim = True)   #(1,10000)
        return signal
    
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index,5]}"       #self.annotations is the pandas dataframe #iloc -> is used to access the location of csv by taking 2 coordinates
        path = os.path.join(self.audio_dir, fold ,self.annotations.iloc[index,0])   #index ->all rows (x coordinate) , 0 ->Oth column (Y coordinate)
        return path
    
    def _get_audio_sample_label (self, index):
        return self.annotations.iloc[index, 6] 
    
    
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
    #Sample Rate -> Quality
    #Num samples -> for same length
    #n_fft -> Frame size 
    #hop_length ->(half the frame size)-> to control the temporal resolution of the spectrogram
    #n_mels -> mel scale for audio recognisation
    
    
    
    
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE , NUM_SAMPLES , device)
    
    print(f"There are {len(usd)} samples in the dataset.")
    
    signal, label = usd[0]
        
    a = 1
    