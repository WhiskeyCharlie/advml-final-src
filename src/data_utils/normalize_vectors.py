from pathlib import Path

import numpy as np
from glob import glob


VECTOR_INPUT_PATH = Path('/home/jon/Documents/AdvML/final_project/data/eng-all/svd/')
VECTOR_OUT_PATH = Path('/home/jon/Documents/AdvML/final_project/data/eng-all/processed_svd/')


def normalize_vector_file(vector_path: str, output_path: str):
    vectors = np.load(vector_path)
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1)[:, np.newaxis])
    np.save(output_path, vectors)


def main():
    for file in glob(str(VECTOR_INPUT_PATH / '*.npy')):
        path = Path(file)
        result_path = str(VECTOR_OUT_PATH / str(path.name))
        normalize_vector_file(file, result_path)


if __name__ == '__main__':
    main()
