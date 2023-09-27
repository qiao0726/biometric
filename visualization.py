import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

def visualize3d_color(X, save_path, labels):
    """ Visualize the data in 3D using PCA and save the figure to the specified path.
    Args:
        X (Tensor): A pytorch tensor of shape (N, D) where N is the number of samples and D is the number of features.
        save_path (str): The path to save the figure to.
    """
    # Convert the tensor to a numpy array
    X = X.detach().numpy()
    
    # Initialize PCA and set the number of components to 3
    pca = PCA(n_components=3)
    # # Fit the PCA model to your data
    # pca.fit(X)
    # Transform the data to the first three principal components
    X_reduced = pca.fit_transform(X)
    
    # Get the unique elements of the tensor
    unique_elements, counts = torch.unique(labels, return_counts=True)
    # How many unique labels
    label_type = unique_elements.shape[0]
    # Create a colormap
    cmap = plt.get_cmap('tab20', label_type)
    
    # Combining 'tab20' and 'tab10' colormaps
    colors = np.concatenate([plt.get_cmap('tab20')(np.arange(20)), 
                            plt.get_cmap('tab20b')(np.arange(20))])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    labels = labels.detach().numpy()
    
    # Plot each data point with color based on its label
    for lb in labels:
        idx = np.where(labels == lb)
        ax.scatter(X_reduced[idx, 0], X_reduced[idx, 1], X_reduced[idx, 2], color=colors[lb], label=lb)
    
    ax.set_title("3D Data Points with 40 Unique Labels")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    
    # Optionally, create a custom legend with label-color mapping
    # This will only show unique labels in the legend
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    new_handles = []
    new_labels = []
    for handle, label in zip(handles, labels_legend):
        if label not in new_labels:
            new_handles.append(handle)
            new_labels.append(label)
    ax.legend(new_handles, new_labels, title="Labels")

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return