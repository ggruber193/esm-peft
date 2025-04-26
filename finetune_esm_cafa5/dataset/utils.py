import datasets
from datasets import Dataset, DatasetDict


def sample_dataset(dataset: Dataset | DatasetDict, n: int | dict[str, int], offset: int | dict[str, int]=0) -> Dataset | DatasetDict:
    if isinstance(dataset, DatasetDict):
        out_dataset = {}
        for key in dataset.keys():
            c_n = n[key]
            c_offset = offset[key]
            out_dataset[key] = dataset[key].select(range(c_offset, c_offset + c_n))
        out_dataset = DatasetDict(out_dataset)
    else:
        out_dataset = dataset.select(range(n))
    return out_dataset


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
