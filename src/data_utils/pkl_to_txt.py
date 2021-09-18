import pickle
from glob import glob
from pathlib import Path


def pkl_file_to_txt_file(pkl_path: str, txt_path: str):
    with open(pkl_path, 'rb') as pkl_file:
        pkl_list_of_words = pickle.load(pkl_file)
    with open(txt_path, 'w') as txt_file:
        print('\n'.join(pkl_list_of_words), file=txt_file)


def main():
    DATA_PATH = Path('/home/jon/Documents/AdvML/final_project/data/eng-all/svd/')
    DATA_PATH_OUT = Path('/home/jon/Documents/AdvML/final_project/data/eng-all/processed_svd/')

    for file in glob(str(DATA_PATH / '*.pkl')):
        path = Path(file)
        result_path = str(DATA_PATH_OUT / str(path.name).replace('.pkl', '.txt'))
        pkl_file_to_txt_file(file, result_path)


if __name__ == '__main__':
    main()
