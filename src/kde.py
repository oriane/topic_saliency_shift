import pandas as pd

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

def plot_kde_compare(topic_winrate, selected_topic, save_folder):
    model_compares = topic_winrate.loc[topic_winrate['model_type'].isin(['sota', 'large', 'ft', 'pt', 'cot'])].copy()
    model_compares['model_type'] = pd.Categorical(
        model_compares['model_type'],
        categories=['pt'] + [m for m in model_compares['model_type'].unique() if m != 'pt'],
        ordered=True
    )
    topic_winrate.loc[topic_winrate.model_name == 'eci-io/climategpt-13b', 'model_type'] = 'climategpt'

    rename_dict = {'pt': 'Base',
                   'ft': 'Post-trained',
                   'large': 'Post-trained large',
                   'sota': 'Leading', 'climategpt': 'ClimateGPT',
                   'cot': 'CoT'}

    selected_winrate = topic_winrate.loc[topic_winrate.model_type.isin(['pt', 'ft', 'climategpt', 'sota', 'cot'])].copy()

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 7))

    # Colors now represent Column 0, Column 1, Column 2
    col_colors = [
        (0, 0.5, 0, 0.2),  # Green for Col 0
        (0.5, 0.5, 0.5, 0.1),  # Grey for Col 1
        (1, 0, 0, 0.2)  # Red for Col 2
    ]

    for idx, (ax, topic) in enumerate(zip(axes.T.ravel(), selected_topic)):
        ax.set_facecolor('white')
        ax.set_zorder(10)

        selected_winrate[f'{topic}_win_rate'] = selected_winrate[topic] / selected_winrate['n']
        for mtype in selected_winrate.model_type.unique():
            selected_winrate.loc[selected_winrate.model_type == mtype][f'{topic}_win_rate'].plot.kde(
                alpha=0.6, label=mtype, ax=ax, linestyle='--'
            )
        ax.set_title(topic)

        # Keeping your specific x-limit logic
        if topic in ['alternative economic system', 'socio-political approaches', 'nuclear']:
            ax.set_xlim([-0.01, 0.06])
        elif topic in ['carbon-pricing', 'community-centric', 'clean transportation']:
            ax.set_xlim([-0.01, 0.3])
        else:
            ax.set_xlim([-0.01, 1])

    plt.tight_layout()
    fig.canvas.draw()

    padding = 0.01

    for col_idx in range(3):
        col_axes = axes[:, col_idx]  # Select the whole column

        # Top-most and Bottom-most positions
        pos_top = col_axes[0].get_position()
        pos_bottom = col_axes[2].get_position()

        left = pos_top.x0 - padding
        bottom = pos_bottom.y0 - padding
        width = (pos_top.x1 - pos_top.x0) + (2 * padding)
        height = (pos_top.y1 - pos_bottom.y0) + (2 * padding)

        rect = Rectangle((left, bottom), width, height,
                         transform=fig.transFigure,
                         facecolor=col_colors[col_idx],
                         edgecolor='none',
                         zorder=0)
        fig.add_artist(rect)

    # Legend handling
    handles, labels = axes[0, 0].get_legend_handles_labels()
    new_labels = [rename_dict.get(l, l) for l in labels]
    fig.legend(handles, new_labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5)

    plt.savefig(f'{save_folder}/density.png', bbox_inches='tight', pad_inches=0.2, transparent=False)
    print(f'KDE plot saved to {save_folder}/density.png')