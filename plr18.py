from pytorch_lightning.core.lightning import LightningModule
import torch
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
import torchmetrics


n_classes = 10

class plr18(LightningModule):
    def __init__(self):
        super(plr18, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, n_classes)
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        
        loss = self.criteria(logits, y)
        acc = torchmetrics.functional.accuracy(logits.softmax(dim=-1), y)
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        loss = loss.unsqueeze(dim=-1)
        return {"loss": loss, "acc": acc}
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        
        val_loss = self.criteria(logits, y)
        val_acc = torchmetrics.functional.accuracy(logits.softmax(dim=-1), y)
        
        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc)
        val_loss = val_loss.unsqueeze(dim=-1)
        return {"val_loss": val_loss, "val_acc": val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss) 
        self.log('avg_val_acc', avg_acc)
        return {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc}
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
#     def train_dataloader(self):
#         return self.train_loader

#     def val_dataloader(self):
#         return self.val_loader