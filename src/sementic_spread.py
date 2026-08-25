import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist


def get_mean_spread(X):
    pdists = pdist(np.stack(X.values), metric='cosine')
    return np.mean(pdists)

def plot_and_save_sementic_spread(to_plot, save_folder):
    sns.ecdfplot(data=to_plot, x='spread', hue='model_type')
    plt.ylabel(r'$F_{\mathcal{T}}(\sigma)$')
    plt.xlabel(r'$\sigma$')
    plt.savefig(f'{save_folder}/sementic_spead.png', bbox_inches='tight', pad_inches=0.1, transparent=True)
    print(f'Sementic spread plot saved at {save_folder}/sementic_spead.png')