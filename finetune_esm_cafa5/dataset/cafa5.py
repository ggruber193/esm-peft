import os
import shutil
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
import tqdm
from datasets import Dataset, NamedSplit, DatasetDict, load_from_disk

from finetune_esm_cafa5.dataset.fasta_dataset import read_sequences, extract_header_info
from finetune_esm_cafa5.dataset.utils import combine_datasets

def extract_header_cafa_train(header: str) -> dict:
    header = ' '.join(header.split(' ')[1:])
    output_dict = extract_header_info(header)
    return output_dict

def extract_header_cafa_test(header: str):
    protein_id = header.split('\t')[0]
    taxon_id = header.split('\t')[1]
    return {"id": protein_id, "OX": taxon_id}


class Cafa5Dataset:
    """
    Downloads the cafa5 competition dataset and processes it to huggingface datasets.
    Make sure you set the kaggle environment variables KAGGLE_USERNAME and KAGGLE_KEY or have a .env file with the envs.
    """
    def __init__(self, data_dir: str | Path, tmp: str | Path = None, cluster_file: str | Path = None, filter_terms: str | Path = None):
        self.data_dir = Path(data_dir)
        self.embedding_dir = self.data_dir / "embeddings"
        self._dataset_path = self.data_dir.joinpath("dataset")
        self._dataset_classification = self.data_dir.joinpath('dataset_classification')
        self._dataset_classification_pos_weight = self._dataset_classification.joinpath('pos_weight.pt')
        self._tmp = tmp

        self._download_dataset()
        dataset = self._create_dataset()
        if cluster_file is not None and 'group' not in dataset["train"].column_names:
            self.add_cluster_groups(cluster_file)

        self.__unique_go_terms = None
        self._term_to_label_mapping: dict[str, int] = None
        if filter_terms:
            with open(filter_terms, 'r') as f:
                terms = f.read().split('\n')
        else:
            terms = None
        self.filter_terms: list[str] | None = terms

    def _download_dataset(self):
        if len(list(self.data_dir.glob("*"))) == 0:
            print("Start downloading dataset")
            import dotenv, zipfile, os
            dotenv.load_dotenv()
            import kaggle
            data_dir = Path(self.data_dir)
            data_dir.mkdir(exist_ok=True, parents=True)
            api = kaggle.KaggleApi()
            api.authenticate()
            cafa5 = "cafa-5-protein-function-prediction"
            dataset_file = data_dir.joinpath(f"{cafa5}.zip")
            api.competition_download_files(cafa5, self.data_dir)
            dataset_zip = zipfile.ZipFile(dataset_file)
            dataset_zip.extractall(self.data_dir)
            dataset_zip.close()
            os.remove(dataset_file)
            print("Finished downloading dataset")

    def _create_train_dataset(self, train_sequences: str | Path, label_file: Path | str):
        def generate_example(fasta_file, terms_dict):
            for example in read_sequences(fasta_file, extract_header_fn=extract_header_cafa_train):
                protein_id = example["id"]
                example["terms"] = terms_dict[protein_id]
                yield example
        df_terms = pd.read_table(label_file)
        df_terms = df_terms.groupby(["EntryID"]).agg({"term": lambda x: list(set(x))}).reset_index()
        terms_dict = dict(zip(df_terms["EntryID"], df_terms["term"]))

        gen_kwargs = {
            "fasta_file": str(train_sequences),
            "terms_dict": terms_dict,
        }
        dataset = Dataset.from_generator(generate_example,
                                         gen_kwargs=gen_kwargs,
                                         split=NamedSplit("train"),
                                         cache_dir=self._tmp)
        return dataset

    def _create_test_dataset(self, test_sequences: str | Path):
        gen_kwargs = {"fasta_file": str(test_sequences), "extract_header_fn": extract_header_cafa_test}
        return Dataset.from_generator(read_sequences,
                                      gen_kwargs=gen_kwargs,
                                      split=NamedSplit("test"),
                                      cache_dir=self._tmp)

    def _create_dataset(self):
        if not self._dataset_path.exists():
            train_file = self.data_dir.joinpath("Train/train_sequences.fasta")
            train_terms_file = self.data_dir.joinpath("Train/train_terms.tsv")
            test_file = self.data_dir.joinpath("Test (Targets)/testsuperset.fasta")

            dataset_train = self._create_train_dataset(train_file, train_terms_file)
            dataset_test = self._create_test_dataset(test_file)
            dataset = DatasetDict({
                "train": dataset_train,
                "test": dataset_test
            })
            dataset.save_to_disk(self._dataset_path, max_shard_size="50MB")
        else:
            dataset = load_from_disk(self._dataset_path)
        return dataset

    def get_dataset_for_mlm(self):
        dataset = load_from_disk(self._dataset_path)
        dataset = combine_datasets(dataset)
        dataset = dataset.train_test_split(train_size=0.8, test_size=0.2, seed=42)
        return dataset

    def get_dataset(self):
        dataset = load_from_disk(self._dataset_path)
        return dataset

    def add_cluster_groups(self, cluster_file: str | Path, sep='\t'):
        dataset = load_from_disk(self._dataset_path)
        df_cluster = pd.read_table(cluster_file, header=None, sep=sep)
        df_cluster.columns = ["query", "target"]
        df_cluster = df_cluster.groupby(["query"]).agg({"target": list}).reset_index(drop=True).reset_index(
            names=["group"]).explode(["target"])
        cluster_dict = dict(zip(df_cluster["target"], df_cluster["group"]))

        tmp_dir = self.data_dir.joinpath("_dataset_tmp")
        dataset["train"] = dataset["train"].map(lambda x: {"group": cluster_dict[x["id"]]})
        dataset["train"].save_to_disk(tmp_dir, max_shard_size="50MB")
        train_path = self._dataset_path.joinpath("train")
        shutil.rmtree(train_path)
        shutil.move(tmp_dir, train_path)

    @property
    def _unique_go_terms(self):
        if self.__unique_go_terms is not None:
            unique_go_terms = self.__unique_go_terms
        else:
            terms_file = self.data_dir.joinpath("Train/train_terms.tsv")
            df_terms = pd.read_csv(terms_file, sep="\t")
            if self.filter_terms is not None:
                df_terms = df_terms[df_terms["term"].isin(self.filter_terms)]
            unique_go_terms = df_terms["term"].unique().tolist()
            self.__unique_go_terms = unique_go_terms
        return unique_go_terms

    @property
    def term_to_label_mapping(self) -> dict[str, int]:
        if self._term_to_label_mapping is None:
            unique_go_terms = self._unique_go_terms
            term_to_label_mapping = {j: i for i, j in enumerate(unique_go_terms)}
            self._term_to_label_mapping = term_to_label_mapping
        else:
            term_to_label_mapping = self._term_to_label_mapping
        return term_to_label_mapping

    @property
    def label_to_term_mapping(self):
        unique_go_terms = self._unique_go_terms
        label_to_term_mapping = {i: j for i, j in enumerate(unique_go_terms)}
        return label_to_term_mapping

    @staticmethod
    def _load_embeddings(embeddings_path: str | Path):
        embeddings_path = Path(embeddings_path)
        is_dataset = len(list(embeddings_path.glob("*.json"))) > 0
        if not is_dataset:
            content = sorted(list(embeddings_path.glob("*")), key=lambda x: x.stem)
            embedding_datasets = [load_from_disk(i) for i in content]
            if isinstance(embedding_datasets[0], DatasetDict):
                keys = embedding_datasets[0].keys()
                dataset = DatasetDict(
                    {i: datasets.concatenate_datasets([j[i] for j in embedding_datasets]) for i in keys})
            else:
                dataset = datasets.concatenate_datasets(embedding_datasets)
        else:
            dataset = load_from_disk(embeddings_path)
        return dataset

    def _onehot_encode_labels(self, labels):
        if isinstance(labels[0], list):
            label_terms = [[self.term_to_label_mapping[i] for i in row if i in self.term_to_label_mapping] for row in labels]
            n_rows = len(label_terms)
            labels = np.zeros((n_rows, len(self._unique_go_terms)))
            for i, row in enumerate(label_terms):
                labels[i, row] = 1
        else:
            label_terms = [self.term_to_label_mapping[i] for i in labels if i in self.term_to_label_mapping]
            labels = np.zeros((len(self._unique_go_terms)))
            labels[label_terms] = 1
        return labels.tolist()

    def _get_pos_weight(self):
        # if not self._dataset_classification_pos_weight.exists():
        dataset_train = load_from_disk(self._dataset_path)["train"]
        pos_weight = np.zeros(len(self._unique_go_terms))
        for row in tqdm.tqdm(dataset_train, desc="Calculating pos weight: "):
            terms = row["terms"]
            for term in terms:
                if term in self.term_to_label_mapping:
                    label = self.term_to_label_mapping[term]
                    pos_weight[label] += 1
        pos_weight = (len(dataset_train) - pos_weight) / pos_weight
        pos_weight = torch.tensor(pos_weight)
        """    torch.save(pos_weight, self._dataset_classification_pos_weight)
        else:
            pos_weight = torch.load(self._dataset_classification_pos_weight, weights_only=True)"""
        return pos_weight

    def get_dataset_for_classification(self, model_embeddings: str, num_proc=8) \
            -> tuple[DatasetDict, int, int, torch.Tensor]:
        # if not self._dataset_classification.exists():
        embeddings_path = self.embedding_dir.joinpath(model_embeddings)
        embedding_dataset = self._load_embeddings(embeddings_path).remove_columns(["id"])
        dataset = load_from_disk(self._dataset_path)
        dataset = datasets.concatenate_datasets([dataset["train"], embedding_dataset["train"]], axis=1)
        dataset = dataset.map(lambda x: {"labels": self._onehot_encode_labels(x["terms"])}, batched=True, batch_size=64, num_proc=num_proc)
        cols = set(dataset.column_names)
        keep_cols = {"embedding", "labels"}
        rem_cols = cols.difference(keep_cols)
        dataset = dataset.remove_columns(list(rem_cols))
        dataset = dataset.train_test_split(train_size=0.8, test_size=0.2, seed=42)
        # dataset.save_to_disk(self._dataset_classification, max_shard_size="50MB")
        pos_weights = self._get_pos_weight()
        """else:
            dataset = load_from_disk(self._dataset_classification)
            pos_weights = self._get_pos_weight()"""

        return dataset, len(dataset["train"]["embedding"][0]), len(self._unique_go_terms), pos_weights


"""pos_weights = dataset.map(lambda x: {"pos_weight": np.array(x["labels"]).sum(axis=0)}, batched=True, batch_size=128, num_proc=6, remove_columns=dataset.column_names)
            pos_weight = None
            with tqdm.tqdm(desc="Calculating pos weight: ", total=len(pos_weights)) as pbar:
                for batch_idx in range(0, len(pos_weights), batch_size):
                    batch = pos_weights[batch_idx:batch_idx + batch_size].sum(axis=0)
                    if pos_weight is None:
                        pos_weight = batch
                    else:
                        pos_weight += batch
                    pbar.update(batch_size)
                pos_weight = (len(dataset) - pos_weight) / pos_weight
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)"""

if __name__ == '__main__':
    dataset = Cafa5Dataset("../../data/cafa5", filter_terms='../../data/cafa5/relevant_labels.txt')

    print(dataset.get_dataset_for_classification("esm2_t6_8M_UR50D"))
