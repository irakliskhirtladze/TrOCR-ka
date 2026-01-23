from ml_training.dataset import prepare_dataset
from ml_training.utils import BASE_DIR


def main():
    dataset_dir = prepare_dataset()
    for item in dataset_dir.iterdir():
        print(item)


if __name__ == "__main__":
    main()
