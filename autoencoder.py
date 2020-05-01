import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from scipy.io import wavfile
import scipy
import librosa
import numpy as np

# mnist_data = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
# mnist_data = list(mnist_data)[:4096]
fs, data = wavfile.read('data/wav/84-121123-0000.wav')
data = data.astype('float32') / 32767  # normalize audio
# print(data.shape)
# print(fs)
# print(data.shape[0]/fs)
# print(len(data))
sample_rate=16000
window_size=0.02
window_stride=0.01
window= scipy.signal.hamming

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
print(spect)
print(spect.shape)



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(model, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5) # <--
    # print(mnist_data[0])
    # train_loader = torch.utils.data.DataLoader(mnist_data,
    #                                            batch_size=batch_size,
    #                                            shuffle=True)
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    return outputs

model = Autoencoder()
max_epochs = 1
outputs = train(model, num_epochs=max_epochs)

# for k in range(0, max_epochs, 5):
# 	plt.figure(figsize=(9, 2))
# 	imgs = outputs[k][1].detach().numpy()
# 	recon = outputs[k][2].detach().numpy()
# 	for i, item in enumerate(imgs):
# 		if i >= 9: break
# 		plt.subplot(2, 9, i + 1)
# 		plt.imshow(item[0])

# 	for i, item in enumerate(recon):
# 		if i >= 9: break
# 		plt.subplot(2, 9, 9 + i + 1)
# 		plt.imshow(item[0])
# plt.show()