import torch
import torchaudio
from cnn import CNNNetwork
from Creating_a_custom_dataset import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE ,SAMPLE_RATE,NUM_SAMPLES

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
    model.eval()   #this method changes how the pytorch model behave --> if turned on certain layers like dropout, normalization ,etc gets off bcoz not needed in evaluation
    with torch.no_grad():           #context manager --> the model doesnt calc any gradience 
        predictions = model(input)            #2D Tensor (1,10) --> [ [0.1, 0.01, ....,0.6] -->sum == 1(bcoz of softmax) ||||| (1)-->no. of samples passed , (10) --> no. of classes that the model tries to predict 
        predicted_index = predictions[0].argmax(0)       
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected
        
    

if __name__ == "__main__":
    
    
    #loading back the model
    cnn = CNNNetwork()
    state_dict = torch.load("feedforwardnet.pth")
    cnn.load_state_dict(state_dict)
    
    
    #loading urban sound validation dataset
    mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate= SAMPLE_RATE , n_fft=1024 , hop_length=512 , n_mels=64)
    
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE , NUM_SAMPLES , "cpu")
    
    
    #get a sample from the Urban sound dataset for inference 
    input , target = usd[0][0] , usd[0][1]   #[batch size , num channels , fr , time]
    
    input.unsqueeze_(0)
    
    
    
    
    #make an inference 
    predicted, expected = predict(cnn, input , target, class_mapping)
    
    
    print(f"Predicted: '{predicted}', expected: '{expected}'")  