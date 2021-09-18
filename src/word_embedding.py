from typing import List, Dict, Set

import numpy as np


class Embedding:
    def __init__(self, matrix: np.array, word_to_idx: Dict[str, int]):
        self.__matrix = matrix
        self.__word_to_idx = word_to_idx

    def get_word_vector_from_model(self, word: str) -> np.array:
        """

        :param word:
        :return:
        """
        if word not in self.__word_to_idx:
            raise KeyError(f'Word "{word}" not in vocabulary')
        temp = self.__word_to_idx[word]
        mat = self.__matrix
        vector = self.__matrix[temp]
        return vector / np.linalg.norm(vector)

    def get_closest_n_vectors(self, vector: np.array, stop_words: Set[str], n=1, whitelist=None):
        """

        :param vector:
        :param stop_words:
        :param n:
        :param whitelist:
        :return:
        """
        check_whitelist = whitelist is not None
        word_distance_pairs = []
        for word in self.__word_to_idx.keys():
            if word in stop_words or (check_whitelist and word not in whitelist):
                continue
            target_vector = self.get_word_vector_from_model(word)
            divisor = np.linalg.norm(vector) * np.linalg.norm(target_vector)
            word_distance_pairs.append((word, np.dot(vector, target_vector) / divisor))
        return sorted(word_distance_pairs, key=lambda x: 1 - x[1])[:n]

    def solve_a_is_to_b_as_c_is_to(self, a: str, b: str, c: str, stop_words=None) -> Dict[str, List[str]]:
        if stop_words is None:
            stop_words = set()
        vec_a = self.get_word_vector_from_model(a)
        vec_b = self.get_word_vector_from_model(b)
        vec_c = self.get_word_vector_from_model(c)
        vec_d = (vec_b - vec_a) + vec_c
        vec_d /= np.linalg.norm(vec_d)
        top_10 = {k: [round(v, 3)] for k, v in self.get_closest_n_vectors(vec_d, stop_words=stop_words, n=10)}
        return top_10
