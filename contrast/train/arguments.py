from transformers import TrainingArguments
    
class RankerArguments(TrainingArguments):
    def __init__(self, 
                 mode : str = None, 
                 group_size : int = 2,
                 **kwargs):
        self.mode = mode
        self.group_size = group_size
        super().__init__(**kwargs)