from contrast.train.trainer import ConstrastTrainer
from transformers import TrainingArguments
from contrast.train.parser import Parser
from contrast.modelling import LoadModel
from contrast.loader import LoadData

def train():
    parser = Parser()
    args = parser.parse_args()
    model = LoadModel(args.model_name_or_path)
    dataset, collator = LoadData(args.dataset_name, args.triples_file, args.teacher_file, args.val_file, args.num_negatives)
    trainer = ConstrastTrainer(model, data, args)
    trainer.train()

if __name__ == "__main__":
    train()