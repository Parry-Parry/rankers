import unittest
from unittest.mock import MagicMock, patch
import torch
from transformers import TrainingArguments
from transformers.trainer_utils import EvalLoopOutput
from datasets import Dataset
import pandas as pd
from collections import defaultdict
from contrast.train.trainer import ContrastTrainer
from contrast.train.loss import LOSSES

class TestContrastTrainer(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock()
        self.tokenizer = MagicMock()
        self.data_collator = MagicMock(tokenizer=self.tokenizer)
        self.args = MagicMock(spec=TrainingArguments)
        self.args.group_size = 2
        self.args.eval_batch_size = 8
        self.args.device = 'cpu'
        self.args.eval_metrics = None
        self.eval_dataset = MagicMock(spec=Dataset)
        self.eval_dataset.qrels = MagicMock()
        self.trainer = ContrastTrainer(
            model=self.model,
            args=self.args,
            data_collator=self.data_collator,
            eval_dataset=self.eval_dataset,
            loss='custom_loss'
        )

    def test_initialization(self):
        self.assertIsInstance(self.trainer.custom_log, defaultdict)
        self.assertEqual(self.trainer.model.config.group_size, 2)
        self.assertEqual(self.trainer.loss, LOSSES['custom_loss']())

    @patch('ir_measures.evaluator')
    def test_compute_metrics(self, mock_evaluator):
        mock_evaluator.return_value.calc_aggregate.return_value = {'RR@10': 0.5}
        result_frame = pd.DataFrame()
        metrics = self.trainer.compute_metrics(result_frame)
        self.assertEqual(metrics, {'RR@10': 0.5})

    @patch('time.time', return_value=1234567890)
    @patch('transformers.integrations.deepspeed.deepspeed_init')
    def test_evaluation_loop(self, mock_deepspeed_init, mock_time):
        self.trainer.is_deepspeed_enabled = False
        self.trainer.is_in_train = False
        self.trainer.accelerator = MagicMock()
        self.trainer.accelerator.prepare_model.return_value = self.model
        self.model.to_pyterrier.return_value.transform.return_value = pd.DataFrame()
        self.trainer.compute_metrics = MagicMock(return_value={'RR@10': 0.5})

        output = self.trainer.evaluation_loop(self.eval_dataset, "Evaluation")

        self.assertIsInstance(output, EvalLoopOutput)
        self.assertEqual(output.metrics, {'eval_RR@10': 0.5})

    @patch('time.time', return_value=1234567890)
    @patch('transformers.trainer_utils.speed_metrics')
    def test_evaluate(self, mock_speed_metrics, mock_time):
        self.trainer.evaluation_loop = MagicMock(return_value=EvalLoopOutput(
            predictions=pd.DataFrame(), label_ids=None, metrics={'eval_RR@10': 0.5}, num_samples=100
        ))
        mock_speed_metrics.return_value = {'eval_speed': 1.0}

        metrics = self.trainer.evaluate()

        self.assertEqual(metrics, {'eval_RR@10': 0.5, 'eval_speed': 1.0})

    def test_maybe_log_save_evaluate(self):
        self.trainer.custom_log = defaultdict(lambda: 10.0)
        self.trainer.state = MagicMock(global_step=10)
        self.trainer.args.gradient_accumulation_steps = 1
        self.trainer.control = MagicMock(should_log=True)
        self.trainer._nested_gather = MagicMock(return_value=torch.tensor([10.0]))

        self.trainer._maybe_log_save_evaluate(0.0, 0.0, self.model, None, 0, None)

        self.trainer.log.assert_called()
        self.assertEqual(self.trainer.custom_log['metric'], 0.0)

    @patch('torch.load')
    def test_load_optimizer_and_scheduler(self, mock_torch_load):
        checkpoint = '/path/to/checkpoint'
        self.trainer.loss = MagicMock()
        self.trainer._load_optimizer_and_scheduler(checkpoint)
        self.trainer.loss.load_state_dict.assert_called_with(mock_torch_load.return_value)

    def test_compute_loss(self):
        inputs = {'input_ids': torch.tensor([[1, 2, 3]])}
        self.model.return_value = (torch.tensor(0.5), None, {'metric': 1.0})
        loss = self.trainer.compute_loss(self.model, inputs)
        self.assertEqual(loss.item(), 0.5)

    @patch('torch.load')
    def test_load_from_checkpoint(self, mock_torch_load):
        resume_from_checkpoint = '/path/to/checkpoint'
        self.trainer._load_from_checkpoint(resume_from_checkpoint)
        self.model.load_state_dict.assert_called_with(mock_torch_load.return_value)

if __name__ == '__main__':
    unittest.main()