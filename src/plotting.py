from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud

sns.set_theme()


def plot_word_cloud(target_word: str, word_to_similarity_dict: Dict[str, float], year: int,
                    fig=None, simple_title=False, title_font_size=50):
    if fig is None:
        plt.figure(figsize=(20, 10))
        fig = plt.axes()
    cloud = WordCloud(background_color="white", width=1200, height=600)
    cloud.generate_from_frequencies(frequencies=word_to_similarity_dict)
    fig.imshow(cloud, aspect='auto')
    fig.axis("off")
    if simple_title:
        fig.set_title(f'{year}s', fontdict={'size': title_font_size})
    else:
        fig.set_title(f'Descriptions Associated With: {target_word.upper()}, Year: {year}',
                      fontdict={'size': title_font_size})

    return fig


def plot_two_progressions(data_pairs: np.array, x_values: List, topics: Tuple[str, ...],
                          x_label='Year', y_label='Positivity', show_plot=False) -> plt.Figure:
    """
    Plots the given data pairs [(y1, y2, ...), ...] where y1s and y2s etc. are first and second functions
    respectively.
    :param show_plot:
    :param data_pairs:
    :param x_values:
    :param topics:
    :param x_label:
    :param y_label:
    :return:
    """
    data_array = np.array(data_pairs, dtype='float64').T
    fig = plt.figure()
    for i, row in enumerate(data_array):
        sns.lineplot(x=x_values, y=row, label=topics[i].upper())
    title_prefix = ' and '.join([x.upper() for x in topics])
    plt.title(f'{title_prefix} sentiment over Time')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_values, x_values, rotation=45)
    plt.legend()
    if show_plot:
        plt.show()
    return fig
