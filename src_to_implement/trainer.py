# PyTorch imports
import torch as t

# Scientific computing imports
import numpy as np
from sklearn.metrics import f1_score

# Progress bar
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=False,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # Reset gradients
        self._optim.zero_grad()
        
        # Forward pass
        out = self._model(x)
        
        # Calculate loss
        loss = self._crit(out, y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self._optim.step()
        
        return loss.item()

    def val_test_step(self, x, y):
        # Forward pass
        out = self._model(x)
        
        # Calculate loss
        loss = self._crit(out, y)
        
        # Get predictions (no need for sigmoid here as it's in the model)
        pred = out
        
        return loss.item(), pred

    def train_epoch(self):
        self._model.train()
        total_loss = 0
        
        # Iterate through training data
        for x, y in tqdm(self._train_dl, desc="Training", leave=False):
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            
            # Perform training step
            loss = self.train_step(x, y)
            total_loss += loss
        
        # Return average loss
        return total_loss / len(self._train_dl)

    def val_test(self):
        self._model.eval()
        total_loss = 0
        predictions = []
        labels = []
        
        # Disable gradient computation
        with t.no_grad():
            for x, y in tqdm(self._val_test_dl, desc="Validating", leave=False):
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                
                # Perform validation step
                loss, pred = self.val_test_step(x, y)
                total_loss += loss
                
                # Store predictions and labels
                predictions.extend(pred.cpu().numpy())
                labels.extend(y.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(predictions) > 0.5  # Convert to binary
        labels = np.array(labels)
        
        # Calculate F1 score for each class
        f1_scores = [
            f1_score(labels[:, i], predictions[:, i]) 
            for i in range(labels.shape[1])
        ]
        
        print(f"Validation F1 Scores - Crack: {f1_scores[0]:.3f}, Inactive: {f1_scores[1]:.3f}")
        
        return total_loss / len(self._val_test_dl)

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        
        # Initialize lists for losses
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        epoch_counter = 0
        
        while True:
            # Check if we should stop by epoch number
            if epochs > 0 and epoch_counter >= epochs:
                break
            
            # Train for one epoch
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.val_test()
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch_counter}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch_counter)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if self._early_stopping_patience > 0 and patience_counter >= self._early_stopping_patience:
                print(f"Early stopping triggered after {epoch_counter + 1} epochs")
                break
            
            epoch_counter += 1
        
        return train_losses, val_losses
                    
        
        
        
