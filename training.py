import os

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from modules.baseblock import ResNet18  # noqa: F401
from modules.dataset import SpectrumDataset
from modules.model import PredictNet

# parameters
train_filepath = "data/train.h5"
dev_filepath = "data/dev.h5"

batch_size = 128
lr = 1e-3
epochs = 50

# set random seed
torch.manual_seed(0)
np.random.seed(0)

# load dataset
train_dataset = SpectrumDataset(train_filepath)
dev_dataset = SpectrumDataset(dev_filepath)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

# load model
save_path = "ckpt/resnet18_f3.pth"
model = PredictNet(ResNet18)

# loss function and optimizer
criterion = MSELoss()
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# move to GPU
model.cuda()
criterion.cuda()

# training
os.makedirs(os.path.dirname(save_path), exist_ok=True)
best_loss = float("inf")
for epoch in trange(epochs, desc="Epoch"):
    model.train()
    for spectra, params in tqdm(train_loader, desc="Train"):
        spectra, params = spectra.cuda(), params.cuda()
        optimizer.zero_grad()
        preds = model(spectra)
        loss = criterion(preds, params)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        dev_loss = 0
        for spectra, params in dev_loader:
            spectra, params = spectra.cuda(), params.cuda()
            preds = model(spectra)
            dev_loss += criterion(preds, params)

    print(f"Epoch {epoch}, dev loss: {dev_loss / len(dev_loader)}")

    # save the best model
    if dev_loss < best_loss:
        best_loss = dev_loss
        torch.save(model.state_dict(), save_path)

# save the final model
torch.save(model.state_dict(), "ckpt/last.pth")
