import torch
import torch.nn as nn
import torch.nn.functional as F

def get_pairwise_distances(embeddings, squared=False):
    """ Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings (Tensor): Tensor of shape (batch_size, embedding_size)
        squared (bool, optional): If True, output is the pairwise squared euclidean distance matrix.
    Returns:
        distances: A tensor of shape (batch_size, batch_size) with the pairwise distances.
    """
    # dot_product.shape = (batch_size, batch_size)
    dot_product = torch.matmul(embeddings, embeddings.t())   
    # On the diagonal is the square of each embedding
    square_norm = torch.diag(dot_product)
    # |a-b|^2 = a^2 - 2ab + b^2
    distances = torch.unsqueeze(square_norm, dim=1) - 2.0 * dot_product + torch.unsqueeze(square_norm, dim=0)
    # Deal with numerical inaccuracies. Set small negatives to zero.
    distances = torch.clamp(distances, min=0.0)
    # If not squared, compute the sqrt, notice there are 0 elements, so add 1e*-16
    if not squared:
        # mask is a matrix with 0s and 1s, 1 means the element is 0
        mask = torch.eq(distances, 0.0).float()
        # Add a small number to the zeros to avoid dividing by zero
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        # Set the elements in mask back to 0
        distances = distances * (1.0 - mask)
    return distances

def get_valid_positive_mask(labels):
    """ Return a 2D mask where mask[a, p] is 1 iff a and p are distinct and have same label.

    Args:
        labels (Tensor): shape = (batch_size, 1)
    Returns:
        mask (Tensor): shape = (batch_size, batch_size), contains only 0s and 1s
    """
    # mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
    mask = torch.eq(labels, labels.transpose(0, 1))
    
    # Set the diagonal to 0 as anchor and positive are not distinct
    mask = mask.float() - torch.eye(labels.size(0)).to(labels.device)
    return mask

def get_valid_negative_mask(labels):
    """ Get a mask for every valid negative (they should have different labels)

    Args:
        labels (Tensor): shape = (batch_size, )

    Returns:
        mask (Tensor): shape = (batch_size, batch_size), contains only 0s and 1s
    """
    # mask is a matrix with 0s and 1s, 1 means the label is identical
    mask = torch.eq(labels, labels.transpose(0, 1))
    # Set the diagonal to 0 as anchor and negative are not distinct
    mask = 1.0 - mask.float()
    return mask

def get_correct_pairs(pairwise_distances, labels):
    total_pairs_num = pairwise_distances.shape[0]
    
    # The 0s in pairwise_distances means the anchor itself, change it to inf
    # Replace all 0 values with inf
    inf_tensor = torch.tensor(float('inf')).to('cuda:7')
    pairwise_distances = torch.where(pairwise_distances == 0.0, inf_tensor, pairwise_distances)
    
    # Get the unique elements of the tensor
    unique_elements, counts = torch.unique(labels, return_counts=True)
    # Get the number of unique elements(anchors with no positives)
    no_positive_anchor_num = torch.sum(counts == 1)
    
    # Get the index of the minimum value of each row
    # min_indices.shape = (batch_size, )
    min_indices = torch.argmin(pairwise_distances, dim=1)
    # The min distance of each anchor
    min_labels = torch.index_select(labels, dim=0, index=min_indices)
    
    correct_pairs_num = torch.sum(torch.eq(labels, min_labels))
    
    return int(correct_pairs_num), int(total_pairs_num), int(no_positive_anchor_num)


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, batch_hard=False, squared=False):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.batch_hard = batch_hard
        self.squared = squared

    def forward_batch_hard(self, embeddings, labels):
        """ With a batch of  embeddings, get the hardest positive and negative for each anchor.

        Args:
            embeddings (Tensor): A batch of embeddings of shape (batch_size, embedding_size)
            labels (Tensor): A (batch_size) tensor containing the labels for all samples in the batch
        Returns:
            triplet_loss (Tensor): The triplet loss
        """
        pairwise_distances = get_pairwise_distances(embeddings, squared=False)
        
        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        valid_positive_mask = get_valid_positive_mask(labels).to(embeddings.device).float()
        # We put any element to 0 where (a, p) is not valid (valid if a != p and label(a) == label(p))
        # anchor-posotive distance==0 means no need to calculate the loss
        # element-wise multiplication
        anchor_positive_dist = torch.mul(valid_positive_mask, pairwise_distances)
        
        # For each anchor, get the hardest positive
        # shape (batch_size, 1)
        # This will return the max values of each row of the input tensor in the given dimension dim.
        # which means the max distance of each (anchor, positive) pair
        hardest_positive_dist = torch.max(anchor_positive_dist, dim=1, keepdim=True)[0]
        #-----------------------------Hardest positive done---------------------------------
        
        
        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        valid_negative_mask = get_valid_negative_mask(labels).to(embeddings.device).float()
        
        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist = torch.max(pairwise_distances, dim=1, keepdim=True)[0]
        invalid_negative_mask = (1.0 - valid_negative_mask)
        # shape (batch_size, batch_size)
        anchor_negative_dist = pairwise_distances + torch.mul(max_anchor_negative_dist, invalid_negative_mask)
        
        # hardest_negative_dist.shape = (batch_size, 1), the min distance of each (anchor, negative) pair
        hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)
        
        
        # Combine biggest distance(a, p) and smallest distance(a, n) into final triplet loss
        # triplet_loss.shape = (batch_size, 1)
        # Each element in triplet_loss is the max(0, d(a, p) - d(a, n) + margin)
        triplet_loss = torch.max(hardest_positive_dist - hardest_negative_dist + self.margin, torch.tensor(0.0))
        
        
        # Get final mean triplet loss
        triplet_loss = torch.mean(triplet_loss)
        
        # # get the number of anchors with both valid positive and negative
        # invalid_positive_mask = (1.0 - valid_positive_mask)
        # invalid_negative_mask = (1.0 - valid_negative_mask)
        
        # The 0s in hardest_positive_dist means the anchor has no valid positive, change it to inf
        # Replace all 0 values with inf
        # hardest_positive_dist = torch.where(hardest_positive_dist == 0.0, torch.tensor(float('inf')).double(), hardest_positive_dist)
        inf_tensor = torch.tensor(float('inf')).to('cuda:7')
        hardest_positive_dist = hardest_positive_dist.float() # Convert this tensor to float just to be sure
        hardest_positive_dist = torch.where(hardest_positive_dist == 0.0, inf_tensor, hardest_positive_dist)

        
        correct_pairs_num, total_pairs_num, no_positive_anchor_num = get_correct_pairs(pairwise_distances, labels)
        
        return triplet_loss, correct_pairs_num, total_pairs_num, no_positive_anchor_num
     

    
    def forward_batch_all(self, embeddings, labels):
        return None
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings (Tensor): A batch of embeddings of shape (batch_size, embedding_size)
            labels (Tensor): A (batch_size) tensor containing the labels for all samples in the batch

        Returns:
            _type_: _description_
        """
        if(self.batch_hard):
            return self.forward_batch_hard(embeddings, labels)
        else:
            return self.forward_batch_all(embeddings, labels)
    
    
    
    
    