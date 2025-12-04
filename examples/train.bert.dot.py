from transformers import HfArgumentParser

from rankers import (
    Dot,
    DotConfig,
    DotDataCollator,
    RankerDataArguments,
    RankerDotArguments,
    RankerTrainer,
    RankerTrainingArguments,
    TestDataset,
    TrainingDataset,
)


def main():
    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_config = DotConfig.from_pretrained(
        model_args.model_name_or_path,
        pooling_type=model_args.pooling_type,
        model_tied=model_args.model_tied,
        use_pooler=model_args.use_pooler,
        inbatch_loss=model_args.inbatch_loss,
    )
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    dataset = TrainingDataset(
        data_args.training_dataset_file,
        group_size=training_args.group_size,
        corpus=data_args.ir_dataset,
        no_positive=data_args.no_positive,
        teacher_file=data_args.teacher_file,
    )

    collate_fn = DotDataCollator(model.tokenizer)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        loss_fn=training_args.loss_fn,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if data_args.test_dataset_file:
        test_dataset = TestDataset(
            data_args.test_data,
            corpus=data_args.test_ir_dataset,
            lazy_load_text=True,
        )
        trainer.evaluate(test_dataset)


if __name__ == "__main__":
    main()
