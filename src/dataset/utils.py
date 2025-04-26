import datasets
from datasets import Dataset, DatasetDict


def sample_dataset(dataset: Dataset | DatasetDict, n: int) -> Dataset | DatasetDict:
    if isinstance(dataset, DatasetDict):
        for key in dataset.keys():
            dataset[key] = dataset[key].select(range(n))
    else:
        dataset = dataset.select(range(n))
    return dataset


def combine_datasets(dataset: DatasetDict):
    for key in dataset.keys():
        c_dataset = dataset[key]
        cols = c_dataset.column_names
        rm_cols = set(cols)
        rm_cols.remove("sequence")
        c_dataset = c_dataset.remove_columns(list(rm_cols))
        dataset[key] = c_dataset
    dataset = datasets.concatenate_datasets([dataset[i] for i in dataset.keys()], axis=0)
    return dataset
