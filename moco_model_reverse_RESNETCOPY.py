import lightly
import pytorch_lightning as pl
import lightly
import torch.nn as nn
import torch
import numpy as np
import copy

from plr18_RESNETCOPY import plr18


"""
Note:
Ive built this custom moco class to override some functions from the base class just because they
do some things that I dont want. For example, in the forward function for the model, I dont 
necesarily want momentum_update() to occur. Ive also built a custom loss function that doesnt update
its -- TODO

"""

class myNTXentLoss(lightly.loss.NTXentLoss):

    """
    only need to override `forward()`. We will add an argument `update_memory_bank`.
    """
    def forward(self,
                out0: torch.Tensor,
                out1: torch.Tensor,
                update_memory_bank=True):
        """Forward pass through Contrastive Cross-Entropy Loss.

        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as 
        negative samples.

            Args:
                out0:
                    Output projections of the first set of transformed images.
                    Shape: (batch_size, embedding_size)
                out1:
                    Output projections of the second set of transformed images.
                    Shape: (batch_size, embedding_size)
                update_memory_bank:
                    By default this is True, which just acts as the original behavior, i.e. 
                    the memory bank is updated iff `out0.requires_grad=True`. Otherwise,
                    dont update memory bank.

            Returns:
                Contrastive Cross Entropy Loss value.

        """

        device = out0.device
        batch_size, _ = out0.shape

        # normalize the output to length 1
        out0 = torch.nn.functional.normalize(out0, dim=1)
        out1 = torch.nn.functional.normalize(out1, dim=1)

        # ask memory bank for negative samples and extend it with out1 if 
        # out1 requires a gradient, otherwise keep the same vectors in the 
        # memory bank (this allows for keeping the memory bank constant e.g.
        # for evaluating the loss on the test set)
        # out1: shape: (batch_size, embedding_size)
        # negatives: shape: (embedding_size, memory_bank_size)
        out1, negatives = \
            super(lightly.loss.NTXentLoss, self).forward(out1, update=out0.requires_grad and update_memory_bank)

        # We use the cosine similarity, which is a dot product (einsum) here,
        # as all vectors are already normalized to unit length.
        # Notation in einsum: n = batch_size, c = embedding_size and k = memory_bank_size.

        if negatives is not None:
            # use negatives from memory bank
            negatives = negatives.to(device)


            # sim_pos is of shape (batch_size, 1) and sim_pos[i] denotes the similarity
            # of the i-th sample in the batch to its positive pair
            sim_pos = torch.einsum('nc,nc->n', out0, out1).unsqueeze(-1)

            # sim_neg is of shape (batch_size, memory_bank_size) and sim_neg[i,j] denotes the similarity
            # of the i-th sample to the j-th negative sample
            sim_neg = torch.einsum('nc,ck->nk', out0, negatives)

            # set the labels to the first "class", i.e. sim_pos,
            # so that it is maximized in relation to sim_neg
            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)


        else:
            # use other samples from batch as negatives
            output = torch.cat((out0, out1), axis=0)

            # the logits are the similarity matrix divided by the temperature
            logits = torch.einsum('nc,mc->nm', output, output) / self.temperature
            # We need to removed the similarities of samples to themselves
            logits = logits[~torch.eye(2*batch_size, dtype=torch.bool, device=out0.device)].view(2*batch_size, -1)

            # The labels point from a sample in out_i to its equivalent in out_(1-i)
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            labels = torch.cat([labels + batch_size - 1, labels])

        loss = self.cross_entropy(logits, labels)

        return loss

class myMoCo(lightly.models.MoCo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                return_features: bool = False,
                do_momentum_update: bool = True):
        """Embeds and projects the input image.

        Performs the momentum update, extracts features with the backbone and 
        applies the projection head to the output space. If both x0 and x1 are
        not None, both will be passed through the backbone and projection head.
        If x1 is None, only x0 will be forwarded.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
            return_features:
                Whether or not to return the intermediate features backbone(x).

        Returns:
            The output projection of x0 and (if x1 is not None) the output
            projection of x1. If return_features is True, the output for each x
            is a tuple (out, f) where f are the features before the projection
            head.

        Examples:
            >>> # single input, single output
            >>> out = model(x) 
            >>> 
            >>> # single input with return_features=True
            >>> out, f = model(x, return_features=True)
            >>>
            >>> # two inputs, two outputs
            >>> out0, out1 = model(x0, x1)
            >>>
            >>> # two inputs, two outputs with return_features=True
            >>> (out0, f0), (out1, f1) = model(x0, x1, return_features=True)

        """
        if do_momentum_update: self._momentum_update(self.m) # SHATZ - added if statement
        
        # forward pass of first input x0
        f0 = self.backbone(x0).flatten(start_dim=1)
        out0 = self.projection_head(f0)

        # append features if requested
        if return_features:
            out0 = (out0, f0)

        # return out0 if x1 is None
        if x1 is None:
            return out0

        # forward pass of second input x1
        with torch.no_grad():

            # shuffle for batchnorm
            if self.batch_shuffle:
                x1, shuffle = self._batch_shuffle(x1)

            # run x1 through momentum encoder
            f1 = self.momentum_backbone(x1).flatten(start_dim=1)
            out1 = self.momentum_projection_head(f1).detach()
        
            # unshuffle for batchnorm
            if self.batch_shuffle:
                f1 = self._batch_unshuffle(f1, shuffle)
                out1 = self._batch_unshuffle(out1, shuffle)

            # append features if requested
            if return_features:
                out1 = (out1, f1)

        return out0, out1


class ReverseMocoModel(pl.LightningModule):
    def __init__(self, memory_bank_size):
        super().__init__()
        
        self.automatic_optimization = False

        # create a ResNet backbone and remove the classification head
        # resnet = lightly.models.ResNetGenerator('resnet-18', 1, num_splits=8)
        resnet = plr18().load_from_checkpoint("./saved_models/resnet_80/epoch=73-val_loss=0.43-val_acc=0.90.ckpt").model
        for p in resnet.parameters():
            p.requires_grad = False

        backbone = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Unflatten(1,(512, 1, 1)),
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco based on ResNet
        #self.resnet_moco = lightly.models.MoCo(backbone, num_ftrs=512, m=0.99, batch_shuffle=True)
        self.resnet_moco = myMoCo(backbone, num_ftrs=512, m=None, batch_shuffle=True)

        # create our loss with the optional memory bank
        self.criterion = myNTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size,
            )

    def automatic_optimization(self):
        return False

    def forward(self, x):
        self.resnet_moco(x, do_momentum_update=False)
        
    def contrastive_loss(self, x0, x1):
        # calculate the contrastive loss for some transformed x -> x0, x1
        # also return grad for each of these
        self.zero_grad()
        x0.requires_grad = True
        x1.requires_grad = True
        y0, y1 = self.resnet_moco(x0, x1, do_momentum_update=False)
        loss = self.criterion(y0, y1, update_memory_bank=False)
        # self.manual_backward(loss)
        loss.backward()
        return x0.grad, x1.grad, loss
    
    def contrastive_loss_nograd(self, x0, x1):
        with torch.no_grad():
            y0, y1 = self.resnet_moco(x0, x1, do_momentum_update=False)
            loss = self.criterion(y0, y1, update_memory_bank=False)
        return loss
        
    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_moco(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()


    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_moco.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]
    
    def toggle_optimizer(self):
        print('toggle')
    
    def untoggle_optimizer(self):
        print('untoggle')

