import argparse
from pathlib import Path
from typing import Set, Tuple, Optional, List

import numpy as np
from matplotlib import pyplot as plt

from plotting import plot_two_progressions, plot_word_cloud
from word_embedding import Embedding

OUTPUTS_PATH = Path('../outputs')
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
POS_PATH = Path('../data/opinion-lexicon-English/positive-words-processed.txt')
NEG_PATH = Path('../data/opinion-lexicon-English/negative-words-processed.txt')
EXPERIMENTS = [
    ('immigrant',),
    ('japanese',),
    ('irish',),
    ('christian', 'jew', 'muslim'),
    ('frenchman', 'german'),
    ('alcohol', 'cigarette'),
    ('gold',),
    ('tobacco', 'morphine', 'alcohol'),
    ('cigar', 'cigarette'),
    ('congress', 'president'),
]


def run_single_word_cloud_experiment(year: int, target_word: str, pos_words: Set[str], neg_words: Set[str],
                                     show_output=True, fig=None, simple_title=False):
    data_path = Path('../data/eng-all/processed_sgns')
    with open(data_path / f'{year}-vocab.txt', 'r') as file:
        words = [x.strip() for x in file.readlines()]
    array = np.load(str(data_path / f'{year}-w.npy'))
    assert len(words) == array.shape[0], f'mismatch on year {year}'

    word_to_idx = dict(zip(map(lambda x: x.strip(), words), range(len(words))))
    embedding = Embedding(array, word_to_idx)
    all_words = set(words)
    all_sentiment_words = pos_words.union(neg_words)
    vector = embedding.get_word_vector_from_model(target_word)
    closest_words = embedding.get_closest_n_vectors(vector, stop_words=all_words - all_sentiment_words, n=100)
    word_to_similarity_dict = {word: dist for word, dist in closest_words}
    fig = plot_word_cloud(target_word, word_to_similarity_dict, year, fig=fig, simple_title=simple_title)
    if show_output:
        plt.show()
    return fig


def run_multiple_word_cloud_experiments(years: List[int], target_word: str, pos_words: Set[str], neg_words: Set[str],
                                        output_path=None, show_plot=True):
    fig, axes = plt.subplots(nrows=len(years), figsize=(20, 10 * len(years)))
    fig.suptitle(f'Descriptions associated with {target_word.upper()} over time.', fontsize=50)
    for i, year in enumerate(years):
        run_single_word_cloud_experiment(year, target_word, pos_words, neg_words, fig=axes[i],
                                         show_output=False, simple_title=True)
    if show_plot:
        plt.show()
    years = '-'.join(map(str, years))
    if output_path is None:
        output_path = OUTPUTS_PATH / f'wordcloud-{target_word}-{years}.png'
    fig.savefig(output_path)


def load_sentiment_words():
    def newline_sep_file_to_set(path: Path) -> Set[str]:
        with open(path, 'r', encoding='iso-8859-1') as file:
            words_set = set(map(lambda x: x.strip(), file.readlines()))
        return words_set

    return newline_sep_file_to_set(POS_PATH), newline_sep_file_to_set(NEG_PATH)


def process_single_year(data_path: Path, year: int, topics: Tuple[str, ...], num_words_to_take: int,
                        pos_words: Set[str], neg_words: Set[str]) -> Optional[Tuple[float, ...]]:
    with open(data_path / f'{year}-vocab.txt', 'r') as file:
        words = [x.strip() for x in file.readlines()]
    array = np.load(str(data_path / f'{year}-w.npy'))
    assert len(words) == array.shape[0], f'mismatch on year {year}'

    word_to_idx = dict(zip(map(lambda x: x.strip(), words), range(len(words))))
    embedding = Embedding(array, word_to_idx)
    all_sentiment_words = pos_words.union(neg_words)
    pos_counts = []

    for topic in topics:
        topic_vec = embedding.get_word_vector_from_model(topic)
        topic_closest_pairs = \
            embedding.get_closest_n_vectors(topic_vec, {topic}, n=num_words_to_take,
                                            whitelist=all_sentiment_words)
        topic_closest_words = [p[0] for p in topic_closest_pairs]
        topic_pos_count = len([w for w in topic_closest_words if w in pos_words]) / num_words_to_take
        if np.isnan(topic_vec).any():
            pos_counts.append(None)
        else:
            pos_counts.append(topic_pos_count)

    return tuple(pos_counts)


def run_single_experiment(pos_words: Set[str], neg_words: Set[str], topics: Tuple[str, ...],
                          num_words_to_take=100, output_path=None, show_plot=True):
    pos_word_data_points = []
    years = list(range(1880, 2000, 10))
    data_path = Path('../data/eng-all/processed_sgns')

    for year in years:
        topic_pos_tuple = process_single_year(data_path, year, topics, num_words_to_take, pos_words, neg_words)
        pos_word_data_points.append(topic_pos_tuple)

    fig = plot_two_progressions(pos_word_data_points, years, topics, show_plot=show_plot)
    topics_str = '-'.join(topics)
    if output_path is None:
        output_path = OUTPUTS_PATH / f'linegraph-{topics_str}-n{num_words_to_take}.png'
    fig.savefig(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Word Embedding Sentiment Extraction Tool')
    subparsers = parser.add_subparsers(dest='command')

    parser_line = subparsers.add_parser('linegraph', help='Draw a line-graph for all years (1880 - 1990)')
    parser_line.add_argument('topics', help='Provide the topic of interest, e.g., "immigrant"', nargs='+')

    parser_cloud = subparsers.add_parser('cloud', help='Generate a word-cloud for certain years')
    parser_cloud.add_argument('topic', help='Provide the topic of interest, e.g., "immigrant"', type=str)
    parser_cloud.add_argument('--years', nargs='+', type=int,
                              help='Any years ending in "0" from 1880 to 1990 (inclusive)')

    parser.add_argument('--out', help='Output file path (will probably clobber existing files)')
    return parser.parse_args()


def run_default_experiments(show_plot=False):
    pos_words, neg_words = load_sentiment_words()
    for topics in EXPERIMENTS:
        run_single_experiment(pos_words, neg_words, topics, show_plot=show_plot)
    run_multiple_word_cloud_experiments([1880, 1940, 1990], 'german', pos_words, neg_words, show_plot=show_plot)
    run_multiple_word_cloud_experiments([1880, 1940, 1990], 'frenchman', pos_words, neg_words, show_plot=show_plot)
    run_multiple_word_cloud_experiments([1880, 1940, 1990], 'japanese', pos_words, neg_words, show_plot=show_plot)
    run_multiple_word_cloud_experiments([1880, 1940, 1990], 'immigrant', pos_words, neg_words, show_plot=show_plot)


def main():
    import sys
    if len(sys.argv) == 1:  # Default mode, no parameters given
        run_default_experiments()
        return

    arguments = parse_args()
    pos_words, neg_words = load_sentiment_words()
    if arguments.command == 'linegraph':
        run_single_experiment(pos_words, neg_words, arguments.topics, output_path=arguments.out, show_plot=False)
    else:
        run_multiple_word_cloud_experiments(arguments.years, arguments.topic, pos_words, neg_words,
                                            output_path=arguments.out, show_plot=False)


if __name__ == '__main__':
    main()
