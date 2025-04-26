import json
import os
import shutil
from pathlib import Path
import argparse

import mlflow
import numpy as np
import torch
from datasets import load_from_disk
from transformers import EsmForMaskedLM, EsmTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer

from peft import LoraConfig, PeftModel

from src.dataset.cafa5 import Cafa5Dataset

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 32,
    "target_modules": ["key", "value"],
    "lora_dropout": 0.1,
    "layers_pattern": "layer",
    "layers_to_transform": [1, 2, 3, 4, 5, 6]
}

TRAINING_PARAMS = {
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "seed": 42,
    "lr_scheduler_type": "cosine_with_restarts",
    "bf16": True,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--tmp', type=Path)
    parser.add_argument('--model-name', type=str, default='esm2_t6_8M_UR50D')
    parser.add_argument('--save-tokenized-dataset', action='store_true')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--total-batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--enable-progressbar', action='store_true', default=False)


    args = parser.parse_args()

    model_ckpt = f"facebook/{args.model_name}"
    batch_size = args.total_batch_size
    device_batch_size = args.batch_size
    epochs = args.epochs
    grad_accum_steps = batch_size // device_batch_size
    enable_progressbar = args.enable_progressbar

    save_tokenized_dataset = args.save_tokenized_dataset

    cpus = args.cpu

    data_dir: Path = args.data_dir
    tmp_dir: Path = args.tmp

    assert batch_size % device_batch_size == 0, f"Total batch size {batch_size} must be divisible by batch size {device_batch_size}"

    TRAINING_PARAMS["epochs"] = epochs
    TRAINING_PARAMS["batch_size"] = batch_size
    TRAINING_PARAMS["grad_accum_steps"] = grad_accum_steps
    TRAINING_PARAMS["pretrained_model"] = model_ckpt

    tokenized_dataset_path = data_dir.joinpath(f"tokenized_dataset_{args.model_name}")
    output_model = data_dir.joinpath(f"models/{model_ckpt}_lora")
    checkpoint_path = output_model.joinpath("ckpt")

    dataset = Cafa5Dataset(data_dir, tmp_dir)

    tokenizer = EsmTokenizer.from_pretrained(model_ckpt)

    if not tokenized_dataset_path.exists():
        dataset = dataset.get_dataset_for_mlm()
        dataset = dataset.map(
            lambda x: tokenizer(x["sequence"], return_attention_mask=False, max_length=1024, truncation=True),
            batched=True, num_proc=cpus, remove_columns=["sequence"]
        )
        if save_tokenized_dataset:
            dataset.save_to_disk(tokenized_dataset_path, max_shard_size="50MB")
    else:
        dataset = load_from_disk(tokenized_dataset_path)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="pt")

    model = EsmForMaskedLM.from_pretrained(model_ckpt, torch_dtype=torch.bfloat16)

    lora_config = LoraConfig(**LORA_CONFIG)

    model = PeftModel(model, lora_config, adapter_name="lora")

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(checkpoint_path),
        eval_strategy="steps",
        eval_steps=500,
        learning_rate=TRAINING_PARAMS["learning_rate"],
        num_train_epochs=epochs,
        weight_decay=TRAINING_PARAMS["weight_decay"],
        lr_scheduler_type=TRAINING_PARAMS["lr_scheduler_type"],
        per_device_train_batch_size=device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        remove_unused_columns=False,
        bf16=TRAINING_PARAMS["bf16"],
        save_safetensors=False,
        seed=TRAINING_PARAMS["seed"],
        save_steps=100,
        group_by_length=True,
        logging_strategy='steps',
        logging_steps=50,
        save_total_limit=2,
        disable_tqdm=enable_progressbar
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    checkpoint_present = len(list(checkpoint_path.glob("*"))) > 0
    for direct in checkpoint_path.glob("*"):
        train_state = direct.joinpath("trainer_state.json")
        if not train_state.exists():
            shutil.rmtree(direct)

    run_file = output_model.joinpath("run_id")

    if run_file.exists():
        with open(run_file, 'r') as f_r:
            run_id = json.loads(f_r.read())["run_id"]
    else:
        run_id = None

    mlflow.set_tracking_uri(f"file://{data_dir.absolute()}/mlruns")
    mlflow.set_experiment(f"Finetune ESM2 CAFA5 Competition")
    with (mlflow.start_run(run_name=model_ckpt, run_id=run_id) as run,
          torch.serialization.safe_globals([
              np._core.multiarray._reconstruct,
              np.ndarray,
              np.dtype,
              np.dtypes.UInt32DType,
          ])):

        if not run_file.exists():
            run_file.write_text(json.dumps({"run_id": run.info.run_id}))

        if not checkpoint_present:
            mlflow.log_params(LORA_CONFIG)
            mlflow.log_params(TRAINING_PARAMS)

        trainer.train(resume_from_checkpoint=checkpoint_present)

    trainer.save_model(str(output_model))
