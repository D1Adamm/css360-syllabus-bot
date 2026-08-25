"""Regression tests for the SFTConfig arguments a QLoRA run actually builds.

The optimizer-step bug lived in the wiring, not the estimator: the estimator
already returned ceil-based steps while SFTConfig was left to derive its own
floored budget. These tests stub the ML stack so run_training() can be executed
without a GPU, and assert on what reaches SFTConfig.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import train_qlora


def _install_ml_stubs(captured: dict) -> dict:
    """Register minimal stand-ins for torch/datasets/peft/transformers/trl."""
    saved = {
        name: sys.modules.get(name)
        for name in ("torch", "datasets", "peft", "transformers", "trl")
    }

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
    )
    sys.modules["torch"] = torch

    datasets = types.ModuleType("datasets")

    class _Dataset:
        def __init__(self, rows):
            self.rows = list(rows)

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

        def map(self, fn):
            return _Dataset([fn(row) for row in self.rows])

        def __len__(self):
            return len(self.rows)

    datasets.Dataset = _Dataset
    sys.modules["datasets"] = datasets

    peft = types.ModuleType("peft")

    class _Model:
        def __init__(self):
            self.config = types.SimpleNamespace(use_cache=True)

        def gradient_checkpointing_enable(self):
            return None

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    peft.LoraConfig = lambda **kwargs: kwargs
    peft.get_peft_model = lambda model, config: model
    peft.prepare_model_for_kbit_training = lambda model: model
    sys.modules["peft"] = peft

    transformers = types.ModuleType("transformers")

    class _Tokenizer:
        pad_token = None
        eos_token = "</s>"

        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    class _TrainerCallback:
        pass

    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _Tokenizer()
    )
    transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _Model()
    )
    transformers.BitsAndBytesConfig = lambda **kwargs: kwargs
    transformers.TrainerCallback = _TrainerCallback
    transformers.set_seed = lambda seed: None
    sys.modules["transformers"] = transformers

    trl = types.ModuleType("trl")

    def _sft_config(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return types.SimpleNamespace(**kwargs)

    class _SFTTrainer:
        def __init__(self, model=None, args=None, train_dataset=None,
                     eval_dataset=None, callbacks=None, **kwargs):
            self.model = model
            self.args = args
            self.callbacks = list(callbacks or [])
            self.state = types.SimpleNamespace(global_step=0)

        def train(self):
            # A correctly configured Trainer runs exactly max_steps steps.
            steps = int(self.args.max_steps)
            for callback in self.callbacks:
                for _ in range(steps):
                    callback.on_step_begin(self.args, self.state, None)
                    callback.on_step_end(self.args, self.state, None)
            self.state.global_step = steps
            return types.SimpleNamespace(metrics={"train_steps": steps})

        def evaluate(self):
            return {"eval_loss": 1.0}

        def save_state(self):
            return None

    trl.SFTConfig = _sft_config
    trl.SFTTrainer = _SFTTrainer
    sys.modules["trl"] = trl
    return saved


def _restore_ml_stubs(saved: dict) -> None:
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class RunConfigRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.captured: dict = {}
        saved = _install_ml_stubs(self.captured)
        self.addCleanup(_restore_ml_stubs, saved)

    def _run(self, *, train_count: int, smoke: bool, epochs: str = "3",
             grad_accum: str = "8", per_device: str = "1", gpus: str = "1"):
        tmp = Path(tempfile.mkdtemp())
        train_file = tmp / "train.jsonl"
        val_file = tmp / "validation.jsonl"
        train_file.write_text(
            "".join(
                json.dumps({"instruction": f"Q{i}?", "response": f"A{i}."}) + "\n"
                for i in range(train_count)
            ),
            encoding="utf-8",
        )
        val_file.write_text(
            "".join(
                json.dumps({"instruction": f"V{i}?", "response": f"B{i}."}) + "\n"
                for i in range(5)
            ),
            encoding="utf-8",
        )
        argv = [
            "--train-file", str(train_file),
            "--validation-file", str(val_file),
            "--output-dir", str(tmp / "out"),
            "--epochs", epochs,
            "--per-device-batch-size", per_device,
            "--gradient-accumulation-steps", grad_accum,
            "--gpu-count", gpus,
        ]
        if smoke:
            argv.append("--smoke-test")
        report = train_qlora.run_training(train_qlora.parse_args(argv))
        return self.captured, report, tmp / "out"

    def test_full_run_37_examples_uses_15_steps(self) -> None:
        # The reported bug: 37 examples / accumulation 8 / 3 epochs stopped at
        # 12 optimizer steps (epoch ~2.43) under Trainer's floored budget.
        config, report, out_dir = self._run(train_count=37, smoke=False)
        self.assertEqual(config["max_steps"], 15)
        self.assertNotEqual(config["max_steps"], 12)
        self.assertEqual(config["num_train_epochs"], 3.0)
        # Effective batch size is unchanged by the fix.
        self.assertEqual(config["per_device_train_batch_size"], 1)
        self.assertEqual(config["gradient_accumulation_steps"], 8)
        self.assertEqual(report["effectiveBatchSize"], 8)
        # Full runs keep per-epoch eval/save.
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertEqual(config["save_strategy"], "epoch")

        self.assertEqual(report["estimatedOptimizerSteps"], 15)
        self.assertEqual(report["intendedOptimizerSteps"], 15)
        self.assertEqual(report["completedSteps"], 15)
        self.assertEqual(report["missingOptimizerSteps"], 0)
        self.assertTrue(report["trainingLengthSatisfied"])
        # 15 steps x batch 8 = 120 sample slots, at least 3 epochs of 37.
        self.assertGreaterEqual(
            report["completedSteps"] * report["effectiveBatchSize"], 37 * 3
        )

        resolved = json.loads(
            (out_dir / "resolved_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(resolved["max_steps"], 15)
        self.assertEqual(resolved["mode"], "full")

    def test_full_run_divisible_dataset_unchanged(self) -> None:
        config, report, _ = self._run(train_count=48, smoke=False)
        self.assertEqual(config["max_steps"], 18)
        self.assertEqual(report["estimatedOptimizerSteps"], 18)
        self.assertEqual(report["completedSteps"], 18)
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertEqual(config["save_strategy"], "epoch")

    def test_smoke_run_behaviour_preserved(self) -> None:
        config, report, out_dir = self._run(train_count=37, smoke=True)
        self.assertEqual(config["max_steps"], 3)
        self.assertEqual(config["eval_strategy"], "steps")
        self.assertEqual(config["save_strategy"], "steps")
        self.assertEqual(config["eval_steps"], 3)
        self.assertEqual(config["save_steps"], 3)
        self.assertEqual(report["mode"], "smoke")
        self.assertEqual(report["trainExampleCount"], 4)
        self.assertEqual(report["validationExampleCount"], 2)
        self.assertEqual(report["completedSteps"], 3)
        self.assertEqual(report["intendedOptimizerSteps"], 3)
        # Smoke still estimates the FULL run from the full dataset.
        self.assertEqual(report["estimatedOptimizerSteps"], 15)
        self.assertIsNotNone(report["estimatedTrainingOnlySeconds"])

        resolved = json.loads(
            (out_dir / "resolved_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(resolved["max_steps"], 3)
        self.assertEqual(resolved["mode"], "smoke")

    def test_full_run_short_of_budget_is_not_silent(self) -> None:
        original = train_qlora.evaluate_training_length

        def _short(*, completed_steps, intended_steps):
            return original(completed_steps=12, intended_steps=intended_steps)

        train_qlora.evaluate_training_length = _short
        self.addCleanup(setattr, train_qlora, "evaluate_training_length", original)
        with self.assertRaises(RuntimeError) as ctx:
            self._run(train_count=37, smoke=False)
        self.assertIn("finished short of its intended length", str(ctx.exception))
        self.assertIn("12/15", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
