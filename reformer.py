import torch
import time
import numpy as np
import scipy
import librosa
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns


class Dataset(Dataset):
    def __init__(self, x, sigma=0.05):
        super(Dataset, self).__init__()
        self.x = x
        self.sigma = sigma

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        clean_x = self.x[index]
        spect_cx = self.parse_audio(clean_x)
        return spect_cx.permute(1,0)

    def parse_audio(self, data):
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


class DataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(DataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


cuda = torch.cuda.is_available()
DEVICE = 'cuda' if cuda else 'cpu'


def _collate_fn(data):
    # data.sort(key=lambda x: x.shape[1], reverse=True)
    # x = []
    # lens = []
    # for d in data:
    #     x += [d[0].permute(1, 0)]
    # clean = pad_sequence(x, batch_first=True, padding_value=0)
    return data[0]


train_x = np.load("adv-medium-to-long.npy", allow_pickle=True)


train_dataset = Dataset(train_x)
# print(train_dataset)
train_loader_args = dict(shuffle=False,
                         batch_size=1,
                         num_workers=0,
                         pin_memory=True,
                         collate_fn=_collate_fn) if cuda \
    else dict(shuffle=True, batch_size=1, collate_fn=_collate_fn)
train_loader =DataLoader(train_dataset, **train_loader_args)
train_clean = np.load("original-medium.npy", allow_pickle=True)
train_clean_dataset = Dataset(train_clean)
# print(train_clean_dataset)
train_clean_loader =DataLoader(train_clean_dataset, **train_loader_args)


class MLP(nn.Module):
    def __init__(self, input_dim=161):
        super(MLP, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Linear(input_dim, 144),
            nn.ReLU(),
            nn.Linear(144, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 144),
            nn.ReLU(),
            nn.Linear(144, input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MLP1(nn.Module):
    def __init__(self, input_dim=161):
        super(MLP1, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


reformer_model = MLP1()
reformer_model.load_state_dict(torch.load("mlp1.pth"))
reformer_model = reformer_model.to(DEVICE)


def test(model):
    model.eval()
    origins = []
    recons = []
    with torch.no_grad():
        for batch_idx, c in enumerate(train_loader):
            c = c.to(DEVICE)
            recon = model(c)
            origins.append(c)
            recons.append(recon)

            torch.cuda.empty_cache()

    return origins, recons

# origins, recons = test(reformer_model)
# origins = np.array(origins)
# recons = np.array(recons)
# print(recons)
# a = origins[0]
# b = recons[0]
#
# sns.heatmap(a.cpu().numpy(), cmap='GnBu')
# plt.show()
# sns.heatmap(b.cpu().numpy(), cmap='GnBu')
# plt.show()

# np.save("advs-short-to-long.npy", origins)
# np.save("recons-short-to-long.npy", recons)
