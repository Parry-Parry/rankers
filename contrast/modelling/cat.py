from transformers import PreTrainedModel, AutoModelForSequenceClassification, AutoConfig

class Cat(PreTrainedModel):
    def __init__(
        self,
        classifier: PreTrainedModel,
        config: AutoConfig,
    ):
        super().__init__(config)
        self.classifier = classifier

    def forward(self, loss, sequences, labels=None):
        """Compute the loss given (pairs, labels)"""
        sequences = {k: v.to(self.classifier.device) for k, v in sequences.items()}
        labels = labels.to(self.classifier.device) if labels is not None else None
        logits = self.classifier(**sequences).logits

        if labels is None: output = loss(logits)
        else: output = loss(logits, labels)
        return output

    def save_pretrained(self, model_dir):
        """Save classifier"""
        self.config.save_pretrained(model_dir)
        self.classifier.save_pretrained(model_dir)
    
    def load_state_dict(self, model_dir):
        """Load state dict from a directory"""
        return self.classifier.load_state_dict(AutoModelForSequenceClassification.from_pretrained(model_dir).state_dict())

    @classmethod
    def from_pretrained(cls, model_dir_or_name : str, num_labels=2):
        """Load classifier from a directory"""
        config = AutoConfig.from_pretrained(model_dir_or_name)
        classifier = AutoModelForSequenceClassification.from_pretrained(model_dir_or_name, num_labels=num_labels)
        return cls(classifier, config)