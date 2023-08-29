import torch.nn as nn
import torch

class CRSEntropyLoss(nn.Module):
    def __init__(self):
        super(CRSEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, embeddings, gesture_type):
        loss = self.loss_fn(embeddings, gesture_type)
        batch_samples_num = gesture_type.shape[0]
        
        embeddings_cls_idx = torch.argmax(embeddings, dim=1)
        gesture_type = torch.argmax(gesture_type, dim=1)
        
        batch_correct = torch.sum(embeddings_cls_idx == gesture_type)
        
        return loss, batch_correct, batch_samples_num