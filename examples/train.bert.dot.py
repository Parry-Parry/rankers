import contrast
from contrast import ( 
                      ContrastArguments, 
                      ContrastTrainer, 
                      ValidationLogger, 
                      seed_everything,
                      )
from contrast.modelling import Dot
from contrast.datasets import TripletDataset, DotDataCollator
from transformers import get_constant_schedule_with_warmup
from torch.optim import AdamW
import wandb
from fire import Fire

def train(
        model_name_or_path : str,
        output_dir : str,
        train_dataset : str,
        val_dataset : str,
        val_topics : str,
        batch_size : int = 16,
        lr : float = 0.00001,
        grad_accum : int = 1,
        warmup_steps : int = 0,
        eval_steps : int = 1000,
        epochs : int = 1,
        wandb_project : str = None,
        seed : int = 42,
    ):
    seed_everything(seed)
    if wandb_project is not None:
        wandb.init(project=wandb_project,)
    

    model = Dot.from_pretrained(model_name_or_path)
    args = ContrastArguments(
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

    dataset = TripletDataset(train_dataset)
    collate_fn = DotDataCollator(model.encoder.tokenizer)

    opt = AdamW(model.parameters(), lr=lr)
    val_logger = ValidationLogger(
        metric = 'ndcg_cut_10',
        ir_dataset = val_dataset,
        val_topics = val_topics,
    )

    trainer = ContrastTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_fn,
        optimizers=(opt, get_constant_schedule_with_warmup(opt, warmup_steps)),
        loss_fn = contrast.loss.ContrastiveLoss(),
        callbacks=[val_logger],
        )
    
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == '__main__':
    Fire(train)