import torch
import torch.nn as nn
from dataset import TouchscreenDataset, TouchscreenSensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


def cosine_distance(embeddings):
    """ Calculate the cosine distance of two vectors
    Args:
        embeddings (torch.Tensor): embeddings.shape = (batch_size, embedding_size)
    Returns:
        torch.Tensor: cosine distance.shape = (batch_size, batch_size)
    """
    embeddings = embeddings.float()
    # cosine distance
    cosine_dist = 1 - F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    # cosine similarity
    # cosine_similarity = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    return cosine_dist

def euclidean_distance(embeddings):
    """ Calculate the euclidean distance of two vectors
    Args:
        embeddings (torch.Tensor): embeddings.shape = (batch_size, embedding_size)
    Returns:
        torch.Tensor: euclidean distance.shape = (batch_size, batch_size)
    """
    embeddings = embeddings.float()

    # Compute the Euclidean distance matrix
    euclidean_dist = torch.cdist(embeddings, embeddings)
    return euclidean_dist


class TestEngine(object):
    def __init__(self, model, testset_file_path, test_type='all', 
                 dist_fn='euclidean', threshold=0.5):
        """ Test the model
        Args:
            model (nn.Module): Model to be tested
            testset_file_path (str): Testset csv file path
            test_type (str, optional): Choose from "all", "gesture_only" and "id_only". Defaults to 'all'.
            dist_fn (str, optional): Choose from "cosine" and "euclidean". Defaults to 'euclidean'.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device)
        
        self.test_type = test_type
        if test_type not in ['all', 'gesture_only', 'id_only']:
            raise Exception(f'No test type named {test_type}')
        elif test_type == 'id_only':
            testset = TouchscreenDataset(csv_file_path=testset_file_path)
            self.test_loader = DataLoader(testset, shuffle=False, batch_size=1)
        else:
            testset = TouchscreenSensorDataset(csv_file_path=testset_file_path)
            self.test_loader = DataLoader(testset, shuffle=False, batch_size=1)
        
        # Choose distance function
        if dist_fn == 'cosine':
            self.dist_fn = cosine_distance
        elif dist_fn == 'euclidean':
            self.dist_fn = euclidean_distance
        else:
            raise Exception(f'No distance function named {dist_fn}')
        
        self.threshold = threshold
    
    def test(self):
        if self.test_type == 'id_only':
            return self.test_id_only()
        elif self.test_type == 'gesture_only':
            return self.test_gesture_only()
        elif self.test_type == 'all':
            return self.test_all()
        else:
            raise Exception(f'No test type named {self.test_type}')
        
    def _test_all(self):
        self.model.eval()
        total_pairs = 0
        correct_pairs = 0
        for batch_idx, batch in enumerate(self.test_loader):
            pass
    
    def _test_gesture_only(self):
        self.model.eval()
        total_pairs = 0
        correct_pairs = 0
        for batch_idx, batch in enumerate(self.test_loader):
            pass
        
    def _test_id_only(self):
        self.model.eval()
        total_positive_pairs = 0
        total_negative_pairs = 0
        total_correct_positive_pairs = 0
        total_correct_negative_pairs = 0
        for batch_idx, batch in enumerate(self.test_loader):
            id, label, hold_time, inter_time, distance, speed, total_time, gesture_type = batch
            id, label = id.to(self.device), label.to(self.device)
            hold_time, inter_time = hold_time.to(self.device), inter_time.to(self.device)
            distance, speed = distance.to(self.device), speed.to(self.device)
            total_time, gesture_type = total_time.to(self.device), gesture_type.to(self.device)
            
            # embeddings.shape = (batch_size, embedding_size)
            embeddings = self.model(None, hold_time, inter_time, 
                                    distance, speed, total_time, gesture_type)
            
            # Compute the distance matrix, shape = (batch_size, batch_size)
            # Compare with label, if the distance is less than threshold, then correct_pairs += 1
            distances = self.dist_fn(embeddings)
            
            
            # Compute valid positive mask
            valid_positive_mask = torch.eq(label.unsqueeze(1), label.unsqueeze(0)).float()
            # Remove the diagonal elements
            valid_positive_mask = valid_positive_mask.float() - torch.eye(valid_positive_mask.shape[0])
            valid_positive_num = valid_positive_mask.sum()
            
            # Compute valid negative mask
            valid_negative_mask = torch.ne(label.unsqueeze(1), label.unsqueeze(0)).float()
            valid_negative_num = valid_negative_mask.sum()
            
            # Mask all invalid pairs
            valid_positive_distances = torch.mul(distances, valid_positive_mask)
            valid_negative_distances = torch.mul(distances, valid_negative_mask)
            
            # Count correct pairs
            correct_positive_pairs = torch.le(valid_positive_distances, self.threshold).sum()
            correct_negative_pairs = torch.gt(valid_negative_distances, self.threshold).sum()
            
            total_positive_pairs += valid_positive_num
            total_negative_pairs += valid_negative_num
            total_correct_positive_pairs += correct_positive_pairs
            total_correct_negative_pairs += correct_negative_pairs
        
        # Compute recall
        recall = total_correct_positive_pairs / total_positive_pairs
        # Compute accuracy
        precision = (total_correct_positive_pairs + total_correct_negative_pairs) / (total_positive_pairs + total_negative_pairs)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return recall, precision, f1_score
    
    
    