# using hubert to extract features from an audio file.

import soundfile as sf
from transformers import AutoProcessor, HubertModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960")
audio_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(DEVICE)

audiofile = 'audio.wav'
audio_input, sr = sf.read(audiofile) # see below to set start and end time.
inputs = processor(audio_input, return_tensors='pt').input_values.to(DEVICE)
with torch.no_grad():
    outputs = audio_model(**inputs)
output_mid = outputs.hidden_states[6] #the 7th layer output. you can check the size of outputs.hidden_states to see the layers.

output_mid.mean(dim=0) #averaging all the frames to make the sequence length to 1

# you can also read the audio files in batch, see the example code of HubertModel: https://huggingface.co/docs/transformers/model_doc/hubert

# as you will set the start and end time for word level, you can use soundfile or librosa to read the desired duration.

import soundfile as sf

audiofile = "audio.wav"
start_frame = 1000 #in frames not seconds
end_frame = 2000 #in frames not seconds

data, sample_rate = sf.read(audiofile, start=start_frame, stop=end_frame)

# you can calculate the number of frames based on the desired time and sample rate, and then set the start_frame and end_frame
print(sample_rate)
start_frame = int(start_time * sample_rate)
end_frame = int(end_time * sample_rate)


import librosa

audiofile = "audio.wav"
start_time = 10.0 #in seconds
duration = 5.0 # in seconds

data, sample_rate = librosa.load(audiofile, sr=None, offset=start_time, duration=duration)
# the sr=None parameter is used to preserve the original sample rate of the audio file