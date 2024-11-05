from rankers import ( 
                      RankerTrainingArguments, 
                      RankerDotArguments,
                      RankerDataArguments,
                      RankerTrainer, 
                      )
from transformers import HfArgumentParser
from rankers.modelling import Dot, DotConfig
from rankers.datasets import TrainingDataset, DotDataCollator
from transformers import get_constant_schedule_with_warmup
from torch.optim import AdamW
import wandb

def main():
    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.wandb_project is not None:
        wandb.init(project=training_args.wandb_project,)
    
    model_config = DotConfig.from_pretrained(model_args.model_name_or_path, pooling_type=model_args.pooling_type, model_tied=model_args.model_tied, use_pooler=model_args.use_pooler)
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    dataset = TrainingDataset(data_args.training_dataset, ir_dataset=data_args.ir_dataset)
    collate_fn = DotDataCollator(model.encoder.tokenizer)

    opt = AdamW(model.parameters(), lr=training_args.lr)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        optimizers=(opt, get_constant_schedule_with_warmup(opt, training_args.warmup_steps)),
        loss_fn = "contrastive",
        )
    
    trainer.train()
    trainer.save_model(training_args.output_dir + "/model")

if __name__ == '__main__':
    main()