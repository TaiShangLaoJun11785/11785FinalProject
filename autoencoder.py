import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.nn.utils.rnn import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from scipy.io import wavfile
import scipy
import librosa
import numpy as np
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(DEVICE)
torch.cuda.empty_cache()


sample_path = 'data/wav/84-121123-0000.wav'


class SpectrogramDataset(Dataset):
	def __init__(self, manifest_filepath):
		"""
		Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
		a comma. Each new line is a different sample. Example below:

		/path/to/audio.wav,/path/to/audio.txt
		...

		:param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
		:param manifest_filepath: Path to manifest csv as describe above
		:param labels: String containing all the possible characters to map to
		:param normalize: Apply standard mean and deviation normalization to audio tensor
		:param speed_volume_perturb(default False): Apply random tempo and gain perturbations
		:param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
		"""
		with open(manifest_filepath) as f:
			ids = f.readlines()
		ids = [x.replace('/content/deepspeech.pytorch/LibriSpeech_dataset/val/', './data/').strip().split(',') for x in ids]
		self.ids = ids
		self.size = len(ids)
		# self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
		super(SpectrogramDataset, self).__init__()

	def __getitem__(self, index):
		sample = self.ids[index]
		audio_path, transcript_path = sample[0], sample[1]
		spect = self.parse_audio(audio_path)
		# transcript = self.parse_transcript(transcript_path)
		return spect

	def parse_transcript(self, transcript_path):
		with open(transcript_path, 'r', encoding='utf8') as transcript_file:
			transcript = transcript_file.read().replace('\n', '')
		transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
		return transcript

	def parse_audio(self, audio_path):
		fs, data = wavfile.read(audio_path)
		data = data.astype('float32') / 32767  # normalize audio
		# print(data.shape)
		# print(fs)
		# print(data.shape[0]/fs)
		# print(len(data))
		sample_rate = 16000
		window_size = 0.02
		window_stride = 0.01
		window = scipy.signal.hamming

		n_fft = int(sample_rate * window_size)
		win_length = n_fft
		hop_length = int(sample_rate * window_stride)
		# STFT
		D = librosa.stft(data, n_fft=n_fft, hop_length=hop_length,
		                 win_length=win_length, window=window)
		spect, phase = librosa.magphase(D)
		# S = log(S+1)
		spect = np.log1p(spect)
		spect = torch.FloatTensor(spect)
		mean = spect.mean()
		std = spect.std()
		spect.add_(-mean)
		spect.div_(std)
		return spect

	def __len__(self):
		return self.size


def _collate_fn(batch):
	batch = [torch.transpose(b,0,1) for b in batch]
	batch = pad_sequence(batch)
	batch = torch.transpose(batch, 0, 1).unsqueeze(1)

	return batch


class AudioDataLoader(DataLoader):
	def __init__(self, *args, **kwargs):
		"""
		Creates a data loader for AudioDatasets.
		"""
		super(AudioDataLoader, self).__init__(*args, **kwargs)
		self.collate_fn = _collate_fn


dataset = SpectrogramDataset('./data/libri_val_manifest.csv')
# sample_audio = dataset.parse_audio(sample_path).unsqueeze(0).unsqueeze(0)
sample_audio = dataset.parse_audio(sample_path).unsqueeze(0).unsqueeze(0)


# print(sample_audio)
# mnist_data = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
# mnist_data = list(mnist_data)[:4096]
# train_loader = torch.utils.data.DataLoader(mnist_data,batch_size=batch_size,shuffle=True)

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(  # like the Composition layer you built
			nn.Conv2d(1, 16, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(16, 32, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 64, 7)
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(64, 32, 7),
			nn.ReLU(),
			nn.ConvTranspose2d(32, 16, 3, padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(16, 1, 3, padding=1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		torch.cuda.empty_cache()
		return x
model = Autoencoder()
model = model.to(DEVICE)
max_epochs = 4
batch_size = 4
train_loader = AudioDataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(model, num_epochs=5, batch_size=16, learning_rate=1e-3):
	torch.manual_seed(42)
	criterion = nn.MSELoss()  # mean square error loss
	optimizer = torch.optim.Adam(model.parameters(),
	                             lr=learning_rate,
	                             weight_decay=1e-5)  # <--
	# print(mnist_data[0])

	# train_loader = torch.utils.data.DataLoader(mnist_data,batch_size=batch_size,shuffle=True)
	outputs = []
	total_size = train_loader.__len__()
	for epoch in range(num_epochs):
		total_loss = 0
		start = time.time()
		for original in train_loader:
			# original = sample_audio
			original = original.to(DEVICE)
			# print(data)
			# print(len(data))
			# print(torch.cuda.memory_allocated())
			recon = model(original)
			# print(torch.cuda.memory_allocated())
			# print(recon.shape)
			loss = criterion(recon, original)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			total_loss+=loss.item()
			del original
			torch.cuda.empty_cache()
		end =time.time()
		print('Epoch:{}, Loss:{:.4f}, Time:{:.4f}'.format(epoch + 1, total_loss/total_size,end-start))
		outputs.append((epoch, original, recon), )
	return outputs

outputs = train(model, num_epochs=max_epochs, batch_size=batch_size)
# recon = model(dataset[0].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach().numpy()
# print(dataset[0])
# # print(recon.shape)
# print(recon)
# output_path = 'output.npy'
# np.save(output_path,16000,recon)

