from transformers import TrainingArguments
    
class ContrastArguments(TrainingArguments):
    def __init__(self, 
                 mode : str = None, 
                 num_negatives : int = 1,
                 **kwargs):
        self.mode = mode
        self.num_negatives = num_negatives
        super().__init__(**kwargs)