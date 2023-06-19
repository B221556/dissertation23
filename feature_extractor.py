# using hubert to extract features from an audio file.

from transformers import AutoProcessor, HubertModel
import soundfile as sf
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
audio_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(DEVICE)

#audiofile = '/Users/alexandrasaliba/OneDrive/Dissertation_Data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav'
audiofile = "/work/tc046/tc046/pins0and0needles/03-01-01-01-01-01-01.wav"
audio_input, sr = sf.read(audiofile) # see below to set start and end time.
inputs = processor(audio_input, return_tensors='pt').input_values.to(DEVICE)

with torch.no_grad():
    #outputs = audio_model(**inputs)
    outputs = audio_model(inputs, output_hidden_states=True)
#print(outputs)

output_last = outputs.last_hidden_state
print("Last layer:\n", output_last)

output_mid = outputs.hidden_states[6] #the 7th layer output. you can check the size of outputs.hidden_states to see the layers.
print("Intermediate layer:\n", output_mid)
output_mid.mean(dim=0) #averaging all the frames to make the sequence length to 1

print("number of hidden layers", len(outputs.hidden_states))
# you can also read the audio files in batch, see the example code of HubertModel: https://huggingface.co/docs/transformers/model_doc/hubert

# as you will set the start and end time for word level, you can use soundfile or librosa to read the desired duration.


#WHAT is this part for?
import soundfile as sf

#audiofile = '/Users/alexandrasaliba/OneDrive/Dissertation_Data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav'
audiofile = "/work/tc046/tc046/pins0and0needles/03-01-01-01-01-01-01.wav"

#start_frame = 1000 #in frames not seconds
#end_frame = 2000 #in frames not seconds

#data, sample_rate = sf.read(audiofile, start=start_frame, stop=end_frame)
data, sample_rate = sf.read(audiofile)

# you can calculate the number of frames based on the desired time and sample rate, and then set the start_frame and end_frame
print("Sampling rate is",sample_rate)
print("data size is", len(data))
print("Duration of sound file is", str(round(len(data)/sample_rate,4)),"seconds.")
#start_frame = int(start_time * sample_rate)
#end_frame = int(end_time * sample_rate)


#import librosa

#audiofile = "audio.wav"
#start_time = 10.0 #in seconds
#duration = 5.0 # in seconds

#data, sample_rate = librosa.load(audiofile, sr=None, offset=start_time, duration=duration)
# the sr=None parameter is used to preserve the original sample rate of the audio file