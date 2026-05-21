import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import bambi as bmb
import arviz as az

def compute_glmm(topic_winrate, topic_columns):
    # only consider models with a pre-trained (pt), fine-tuned (ft) and fine-tuned large (large) version
    ntype = topic_winrate.groupby('model_family').model_type.nunique()
    complete_family = ntype[ntype >= 3].index

    complete_family_df = topic_winrate.loc[(topic_winrate.model_family.isin(complete_family)) &
                                       (topic_winrate['model_type'].isin(['ft', 'pt', 'large']))].copy()
    # Convert to categorical with the baseline as the first element
    baseline_model = 'pt'  # comparing both version to baseline, ignoring that some model are from the same family

    complete_family_df['model_type'] = pd.Categorical(
        complete_family_df['model_type'],
        categories=[baseline_model] + [m for m in complete_family_df['model_type'].unique() if m != baseline_model],
        ordered=True
    )

    all_summaries = []

    for topic in tqdm(topic_columns, desc='Compute GLMM'):
        # Filter data and Fit Model
        sub_df = complete_family_df[['model_name', 'qid', 'model_type', 'model_family', topic, 'n']].rename(
            columns={topic: 'topic_count'}).copy()
        model = bmb.Model("prop(topic_count, n) ~ model_type + (1 | model_family) + (1 | qid)", data=sub_df,
                          family="binomial")
        idata = model.fit(draws=2000, tune=1000, progressbar=False)
        summary = az.summary(idata, var_names=["model_type", "1|model_family_sigma"], ci_kind="hdi", ci_prob=0.97)
        summary['topic'] = topic
        summary = summary.reset_index().rename(columns={'index': 'model_type'})

        # Clean model names if needed (e.g., "model[GPT-4]" -> "GPT-4")
        # summary['model_type'] = summary['model_type'].str.extract(r'(.∗?)(.*?)')

        all_summaries.append(summary)

    df = pd.concat(all_summaries)
    df['mean'] = df['mean'].astype('float')
    df['hdi97_lb'] = df['hdi97_lb'].astype('float')
    df['hdi97_ub'] = df['hdi97_ub'].astype('float')
    return df

def plot_topic_shift_full(df_sorted, save_folder):
    fig, ax = plt.subplots(figsize=(12, 15))
    # 2. Shading and Labels
    # Subtle background zones
    ax.axvspan(df_sorted["hdi97_lb"].min() - 0.2, 0, color='#fff4f2', alpha=0.5)  # Soft Red area
    ax.axvspan(0, df_sorted["hdi97_ub"].max() + 0.2, color='#d1ffbd', alpha=0.5)  # Soft Green area
    ax.axvline(x=0, color='#0504aa', linestyle='--', linewidth=1, label="No Effect", zorder=4)

    # Headers for the sides
    ax.text(0.15, 1.01, '← less likely than base', transform=ax.transAxes,
            color='#be0119', fontsize=11, ha='left')
    ax.text(0.85, 1.01, 'more likely than base →', transform=ax.transAxes,
            color='#40a368', fontsize=11, ha='right')

    for t, c, l in [('ft', '#9a0eea', 'post-trained'), ('large', '#029386', 'post-trained large')]:
        df_tmp = df_sorted.loc[df_sorted["model_type"].str.contains(t)]
        coef_error = [df_tmp["mean"] - df_tmp["hdi97_lb"],
                      df_tmp["hdi97_ub"] - df_tmp["mean"]]
        ax.errorbar(
            x=df_tmp["mean"],
            y=df_tmp["topic"],
            xerr=coef_error,
            fmt='o',
            color=c,  # Hide the default marker
            ecolor=c,  # Apply our list of two colors to the bars
            capsize=3,
            markersize=5,
            zorder=3,
            label=l
        )

    # 5. Dynamic Y-Axis Label Coloring
    # This colors the text based on statistical significance (HDI excludes 0)
    plt.draw()  # Required to populate tick labels
    yticks = ax.get_yticklabels()
    df_tmp = df_sorted.loc[df_sorted["model_type"].str.contains('ft')]
    for i, (lower, upper) in enumerate(zip(df_tmp["hdi97_lb"], df_tmp["hdi97_ub"])):
        if upper < 0:
            yticks[i].set_color('#be0119')  # Decreased (Red)
        elif lower > 0:
            yticks[i].set_color('#40a368')  # Increased (Green)

    # Final Polish
    ax.set_xlabel("Log-Odds Ratio", fontsize=12)
    ax.set_xlim(df_sorted["hdi97_lb"].min() - 0.2, df_sorted["hdi97_ub"].max() + 0.2)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_folder}/glmm_large.png', bbox_inches='tight', pad_inches=0.5,
                transparent=True)
    print(f'Full glmm plot saved at {save_folder}/glmm_large.png')

def plot_topic_shift_small(df_sorted, save_folder):
    # Sort by the fine-tuning order and keep the top, bottom and middle 3 topics
    ft_df = df_sorted.loc[df_sorted.model_type == 'model_type[ft]']
    mid = int(len(ft_df['topic'].unique()) / 2)
    selected_topic = ft_df.sort_values('mean').iloc[[-1, -2, -3, mid + 1, mid, mid - 1, 2, 1, 0]].topic.values

    df_small = df_sorted.loc[df_sorted.topic.isin(selected_topic)]

    fig, ax = plt.subplots(figsize=(8, 4))

    # 2. Shading and Labels
    # Subtle background zones
    ax.axvspan(df_small["hdi97_lb"].min() - 0.2, 0, color='#fff4f2', alpha=0.5)  # Soft Red area
    ax.axvspan(0, df_small["hdi97_ub"].max() + 0.2, color='#d1ffbd', alpha=0.5)  # Soft Green area
    ax.axvline(x=0, color='#0504aa', linestyle='--', linewidth=1, label="No Effect", zorder=4)

    # Headers for the sides
    ax.text(0.06, 1.01, '← less likely than base', transform=ax.transAxes,
            color='#be0119', fontsize=11, ha='left')
    ax.text(0.85, 1.01, 'more likely than base →', transform=ax.transAxes,
            color='#40a368', fontsize=11, ha='right')


    for t, c, l in [('ft', '#9a0eea', 'post-trained'), ('large', '#029386', 'post-trained large')]:
        df_tmp = df_small.loc[df_small["model_type"].str.contains(t)]
        coef_error = [df_tmp["mean"] - df_tmp["hdi97_lb"],
                      df_tmp["hdi97_ub"] - df_tmp["mean"]]
        ax.errorbar(
            x=df_tmp["mean"],
            y=df_tmp["topic"],
            xerr=coef_error,
            fmt='o',
            color=c,  # Hide the default marker
            ecolor=c,  # Apply our list of two colors to the bars
            capsize=3,
            markersize=5,
            zorder=3,
            label=l,
            alpha=0.8
        )

    # 5. Dynamic Y-Axis Label Coloring
    # This colors the text based on statistical significance (HDI excludes 0)
    plt.draw()  # Required to populate tick labels
    yticks = ax.get_yticklabels()
    df_tmp = df_small.loc[df_small["model_type"].str.contains('ft')]

    for i, (lower, upper) in enumerate(zip(df_tmp["hdi97_lb"], df_tmp["hdi97_ub"])):
        if upper < 0:
            yticks[i].set_color('#be0119')  # Decreased (Red)
        elif lower > 0:
            yticks[i].set_color('#40a368')  # Increased (Green)

    # Final Polish
    ax.set_xlabel("Log-Odds Ratio", fontsize=12)
    ax.set_xlim(df_small["hdi97_lb"].min() - 0.2, df_small["hdi97_ub"].max() + 0.2)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_folder}/glmm_small.png', bbox_inches='tight', pad_inches=0.5,
                transparent=True)
    print(f'Filtered glmm plot saved at {save_folder}/glmm_small.png')
    return selected_topic

