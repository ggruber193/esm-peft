from pathlib import Path
import re
from typing import Callable

from datasets import Dataset



def extract_header_info(header: str):
    id_part = header.split(' ')[0]

    origin = id_part.split('|')[0]
    uniprot_id = id_part.split('|')[1]
    gene_id = id_part.split('|')[2]

    gene_name_groups = re.search(r" ([^=]+)(?!\S)", header).groups()
    gene_name = gene_name_groups[0] if len(gene_name_groups) > 0 else ""

    key_value_info = re.findall(r"(\S{2})=([^=]+)(?=\S{2}=|$)", header)
    key_value_info = {key.strip(): value.strip() for key, value in key_value_info}
    for i in ("OS", "OX", "GN", "PE", "SV"):
        if i not in key_value_info:
            key_value_info[i] = ""

    output_dict = {"id": uniprot_id, "origin": origin, "gene_name": gene_name, "gene_id": gene_id} | key_value_info
    return output_dict

def read_sequences(fasta_file: str, extract_header_fn: Callable[[str], dict] = extract_header_info):
    with open(fasta_file) as f_r:
        buffer = []
        header_info = {}
        for line in f_r:
            if line.startswith(">"):
                if len(header_info) > 0:
                    output = {"sequence": ''.join(buffer)} | header_info
                    yield output
                    buffer = []
                header = line.lstrip(">")
                header_info = extract_header_fn(header)
            else:
                buffer.append(line.strip())
        output = {"sequence": ''.join(buffer)} | header_info
        yield output

def create_dataset(fasta_file: str | Path, output_dir: str | Path):
    dataset = Dataset.from_generator(read_sequences, gen_kwargs={"fasta_file": str(fasta_file)})
    dataset.save_to_disk(output_dir, max_shard_size="50MB")


if __name__ == '__main__':
    input_file = "../data/uniprotkb_go_0043190_AND_reviewed_true_2025_04_25.fasta"
    output_dir = "../data/uniprotkb_go_0043190_AND_reviewed_true_2025_04_25.dataset"
    create_dataset(input_file, output_dir)
