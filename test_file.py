import medmnist
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str,
                    default=str(Path.joinpath(Path(__file__).parent, 'data')))


if __name__ == "__main__":

    args = parser.parse_args()
    data_path = Path(args.input_dir)

    dataset_name = "pathmnist"
    info = medmnist.INFO[dataset_name]
    print(info)

    train_set = medmnist.PathMNIST(
        root=data_path, split='train', download=True)
    test_set = medmnist.PathMNIST(root=data_path, split='test', download=True)

    print(len(train_set.imgs))
    print(len(train_set.labels))
