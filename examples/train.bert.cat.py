from transformers import HfArgumentParser

from rankers import (
    Cat,
    CatDataCollator,
    RankerDataArguments,
    RankerModelArguments,
    RankerTrainer,
    RankerTrainingArguments,
    TestDataset,
    TrainingDataset,
)


def main():
    parser = HfArgumentParser(
        (RankerModelArguments, RankerDataArguments, RankerTrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.test_dataset_file:
        from ir_measures import nDCG

        training_args.eval_ir_metrics = [nDCG @ 10]

    model = Cat.from_pretrained(model_args.model_name_or_path)

    dataset = TrainingDataset(
        data_args.training_dataset_file,
        group_size=training_args.group_size,
        corpus=data_args.ir_dataset,
        no_positive=data_args.no_positive,
        teacher_file=data_args.teacher_file,
    )
    collate_fn = CatDataCollator(model.tokenizer)

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
