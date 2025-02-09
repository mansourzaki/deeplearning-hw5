# Add these at the very top of train.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Standard library imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# PyTorch imports
import torch as t

# Local imports
from data import ChallengeDataset, train_test_split  # Import train_test_split from data
from trainer import Trainer
import model as net  # Rename the import to avoid conflict

# Use the function
data = pd.read_csv('data.csv', delimiter=';')
train_data, val_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dataset = ChallengeDataset(train_data, mode='train')
val_dataset = ChallengeDataset(val_data, mode='val')

# Create data loaders
train_loader = t.utils.data.DataLoader(
    train_dataset,
    batch_size=16,  # Adjust batch size as needed
    shuffle=True,   # Shuffle training data
    num_workers=4   # Number of parallel workers for data loading
)

val_loader = t.utils.data.DataLoader(
    val_dataset,
    batch_size=16,  # Same batch size as training
    shuffle=False,  # No need to shuffle validation data
    num_workers=4   # Number of parallel workers
)

# create an instance of our ResNet model
model = net.ResNet()  # Use the renamed import

# set up a suitable loss criterion
criterion = t.nn.BCELoss()  # Use BCELoss since we have sigmoid in the model

# set up the optimizer (see t.optim)
optimizer = t.optim.Adam(model.parameters(), lr=1e-3)

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(
    model=model,
    crit=criterion,
    optim=optimizer,
    train_dl=train_loader,
    val_test_dl=val_loader,
    cuda=True,  # Use GPU if available
    early_stopping_patience=5  # Stop if validation loss doesn't improve for 5 epochs
)

# go, go, go... call fit on trainer
res = trainer.fit(epochs=50)  # Train for maximum 50 epochs

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')