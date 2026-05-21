import argparse

import os

from src.sementic_spread import plot_and_save_sementic_spread, get_mean_spread
from src.utils.data_handling import load_full_df

def measure_homogenisation(folder_path, save_folder, model_info_path, keywords_path):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    full_df = load_full_df(folder_path, model_info_path, keywords_path, include_kw_embeddings=True)
    print('Computing Sementic Spread...')
    plot_semantic_spread(full_df, save_folder)

def plot_semantic_spread(full_df, save_folder):
    res_kw = full_df.groupby(['model_name', 'qid']).agg(spread=('emb', get_mean_spread), model_type=('model_type', 'first')).reset_index()
    res_kw.model_type = res_kw.model_type.map({'pt': 'Base',
                                               'ft': 'Post-trained',
                                               'large': 'Post-trained large',
                                               'sota': 'Leading', 'climategpt': 'ClimateGPT',
                                               'cot': 'CoT'})
    to_plot = res_kw.loc[~res_kw.model_type.isna()]
    plot_and_save_sementic_spread(to_plot, save_folder)
    return True


def main():

    parser = argparse.ArgumentParser(description="Measure homogenisation")

    parser.add_argument(
        "--folder_path",
        type=str,
        help="Folder with all keywords data",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="data/generated",
        help="Where to save output embeddings",
    )
    parser.add_argument(
        "--model_info_path",
        type=str,
        default='data/model_info.json',
        help="Path to model information",
    )
    parser.add_argument(
        "--keywords_path",
        type=str,
        help="Path to teh pkl file of keywords embeddings and labels",
    )
    args = parser.parse_args()

    measure_homogenisation(args.folder_path, args.save_folder, args.model_info_path, args.keywords_path)

if __name__ == "__main__":
    main()

