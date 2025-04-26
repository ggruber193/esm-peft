import os
import shutil
from pathlib import Path
import pandas as pd
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
    def __init__(self, data_dir: str | Path, tmp: str | Path = None, cluster_file: str | Path = None):
        self.data_dir = Path(data_dir)
        self.embedding_dir = self.data_dir / "embeddings"
        self._dataset_path = self.data_dir.joinpath("dataset")
        self._tmp = tmp

        self._download_dataset()
        dataset = self._create_dataset()
        if cluster_file is not None and 'group' not in dataset["train"].column_names:
            self.add_cluster_groups(cluster_file)

        self._unique_go_terms = None

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

    def _get_unique_go_terms(self):
        if self._unique_go_terms is not None:
            unique_go_terms = self._unique_go_terms
        else:
            terms_file = self.data_dir.joinpath("Train/train_terms.tsv")
            df_terms = pd.read_csv(terms_file, sep="\t")
            unique_go_terms = df_terms["term"].unique().tolist()
            self._unique_go_terms = unique_go_terms
        return unique_go_terms

    @property
    def term_to_label_mapping(self):
        unique_go_terms = self._get_unique_go_terms()
        term_to_label_mapping = {j: i for i, j in enumerate(unique_go_terms)}
        return term_to_label_mapping

    @property
    def label_to_term_mapping(self):
        unique_go_terms = self._get_unique_go_terms()
        label_to_term_mapping = {i: j for i, j in enumerate(unique_go_terms)}
        return label_to_term_mapping

    def get_dataset_for_classification(self):
        dataset = load_from_disk(self._dataset_path)
        dataset = dataset["train"]
        dataset = dataset.train_test_split(train_size=0.8, test_size=0.2)




if __name__ == '__main__':
    dataset = Cafa5Dataset("../../data/cafa5", cluster_file='../../data/cafa5/Train/train_sequences_cluster.tsv')

    print(dataset.get_dataset())
