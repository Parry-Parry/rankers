import argparse
from transformers import TrainingArguments

def main(triples_file : str, 
        teacher_file : str,
        dataset_name : str, 
        out_dir : str, 
        model_name_or_path : str = 'google/electra-base-discriminator',
        val_file : str = None,
        max_epochs : int = 1, 
        batch_size : int = 16, 
        num_negatives : int = 1,
        lr : float = 0.00001, 
        grad_accum : int = 1,
        warmup_steps : int = 0,
        min_train_steps : int = 50000,
        wandb_project=None,
        mode : str = 'std',
        early_patience : str = 30,
        early_check : str = 4000,
        rank : int = None,
        fp16 : bool = False,):
    

parser = argparse.ArgumentParser(description='Train a contrastive model')
class ContrastArguments():
    x = 1