import argparse

import os

from src.glmm import compute_glmm, plot_topic_shift_full, plot_topic_shift_small
from src.kde import plot_kde_compare
from src.utils.data_handling import load_full_df, get_topic_winrate

def measure_topic_shift(folder_path, save_folder, model_info_path, keywords_path):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    full_df = load_full_df(folder_path, model_info_path, keywords_path, include_kw_embeddings=False)
    topic_winrate, topic_columns = get_topic_winrate(full_df)

    print('Compute and plot Bayesian Generalized Linear Mixed Models...')
    summaries_df =  compute_glmm(topic_winrate, topic_columns)
    selected_topic = plot_glmm(summaries_df, save_folder)

    print('Compare and plot against Sota and Mitigation ...')
    plot_kde_compare(topic_winrate, selected_topic, save_folder)


def plot_glmm(summaries_df, save_folder):
    df_sorted = summaries_df.loc[summaries_df.model_type.str.contains('model_type')].sort_values("mean")
    plot_topic_shift_full(df_sorted, save_folder)
    selected_topic = plot_topic_shift_small(df_sorted, save_folder)
    return selected_topic


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

    measure_topic_shift(args.folder_path, args.save_folder, args.model_info_path, args.keywords_path)

if __name__ == "__main__":
    main()

