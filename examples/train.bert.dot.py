import rankers
from rankers import ( 
                      RankerArguments, 
                      RankerTrainer, 
                      seed_everything,
                      )
from rankers.modelling import Dot
from rankers.datasets import TrainingDataset, DotDataCollator
from transformers import get_constant_schedule_with_warmup
from torch.optim import AdamW
import wandb
from fire import Fire

def train(
        model_name_or_path : str, # Huggingface model name or path to model
        ir_dataset : str, # Path to the IR dataset
        output_dir : str, # Where to save the model and checkpoints
        train_dataset : str, # The path to the training dataset
        batch_size : int = 16, # per device batch size
        lr : float = 0.00001, # learning rate
        grad_accum : int = 1, # gradient accumulation steps (used to increase the effective batch size)
        warmup_steps : int = 0, # number of warmup steps for the scheduler
        eval_steps : int = 1000, # number of steps between evaluations
        epochs : int = 1, # number of training epochs
        wandb_project : str = None, # wandb project name
        seed : int = 42, # random seed
    ):
    seed_everything(seed)
    if wandb_project is not None:
        wandb.init(project=wandb_project,)
    
    model = Dot.from_pretrained(model_name_or_path)
    args = RankerArguments(
        output_dir = output_dir,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum,
        learning_rate = lr,
        warmup_steps = warmup_steps,
        num_train_epochs = epochs,
        eval_steps = eval_steps,
        seed = seed,
        wandb_project = wandb_project,
        report_to='wandb',
    )

    dataset = TrainingDataset(train_dataset, ir_dataset=ir_dataset)
    collate_fn = DotDataCollator(model.encoder.tokenizer)

    opt = AdamW(model.parameters(), lr=lr)

    trainer = RankerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_fn,
        optimizers=(opt, get_constant_schedule_with_warmup(opt, warmup_steps)),
        loss_fn = "rankersive",
        )
    
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == '__main__':
    Fire(train)