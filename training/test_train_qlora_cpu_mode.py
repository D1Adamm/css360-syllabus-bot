"""Tests for the explicit CPU device mode of train_qlora.py.

The ML stack is stubbed, as in test_train_qlora_run_config.py, so these run
anywhere. What they pin down:

  - device selection is explicit (--cpu) and never inferred from the machine;
  - the CPU configuration: NF4 4-bit, float32 compute, no fp16/bf16, batch 1,
    a dict device map of "cpu", no CUDA requirement, no CUDA use even if one
    exists;
  - the CUDA configuration is exactly what it was before CPU mode existed;
  - both modes train on byte-identical chat-formatted text;
  - resolved_config.json and runtime-report.json carry the device, dtype,
    thread, timing and peak-memory facts, and the adapter lands where the
    reporter and the GGUF conversion expect it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import train_qlora


def _install_ml_stubs(captured: dict, *, cuda_available: bool, bf16_supported: bool,
                      sft_config_class=None) -> dict:
    saved = {
        name: sys.modules.get(name)
        for name in ("torch", "datasets", "peft", "transformers", "trl")
    }

    torch = types.ModuleType("torch")
    torch.__version__ = "0.0-stub"
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        is_bf16_supported=lambda: bf16_supported,
    )
    threads = {"value": 8}

    def set_num_threads(count):
        captured["set_num_threads"] = count
        threads["value"] = count

    torch.set_num_threads = set_num_threads
    torch.get_num_threads = lambda: threads["value"]
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
            self.gradient_checkpointing = False

        def gradient_checkpointing_enable(self):
            self.gradient_checkpointing = True
            captured["gradient_checkpointing_enabled"] = True

        def save_pretrained(self, path):
            target = Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "adapter_config.json").write_text("{}", encoding="utf-8")
            (target / "adapter_model.safetensors").write_bytes(b"stub")

    def _lora_config(**kwargs):
        captured["lora"] = kwargs
        return kwargs

    peft.LoraConfig = _lora_config
    peft.get_peft_model = lambda model, config: model
    peft.prepare_model_for_kbit_training = lambda model: model
    sys.modules["peft"] = peft

    transformers = types.ModuleType("transformers")

    class _Tokenizer:
        pad_token = None
        eos_token = "</s>"

        def apply_chat_template(self, messages, **kwargs):
            captured.setdefault("chat_template_calls", []).append(
                {"messages": messages, "kwargs": kwargs}
            )
            return "<chat>" + json.dumps(messages) + "</chat>"

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    class _TrainerCallback:
        pass

    def _from_pretrained_model(model_id, **kwargs):
        captured["from_pretrained"] = {"model_id": model_id, **kwargs}
        return _Model()

    def _bnb_config(**kwargs):
        captured["bnb"] = kwargs
        return kwargs

    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _Tokenizer()
    )
    transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=_from_pretrained_model
    )
    transformers.BitsAndBytesConfig = _bnb_config
    transformers.TrainerCallback = _TrainerCallback
    transformers.set_seed = lambda seed: None
    sys.modules["transformers"] = transformers

    trl = types.ModuleType("trl")

    def _sft_config(**kwargs):
        captured["sft"] = dict(kwargs)
        return types.SimpleNamespace(**kwargs)

    class _SFTTrainer:
        def __init__(self, model=None, args=None, train_dataset=None,
                     eval_dataset=None, callbacks=None, **kwargs):
            self.model = model
            self.args = args
            self.callbacks = list(callbacks or [])
            self.state = types.SimpleNamespace(global_step=0)
            captured["train_texts"] = [row["text"] for row in train_dataset.rows]
            captured["eval_texts"] = (
                [row["text"] for row in eval_dataset.rows] if eval_dataset else []
            )

        def train(self):
            steps = int(self.args.max_steps)
            for callback in self.callbacks:
                for _ in range(steps):
                    callback.on_step_begin(self.args, self.state, None)
                    callback.on_step_end(self.args, self.state, None)
            self.state.global_step = steps
            return types.SimpleNamespace(metrics={"train_steps": steps, "train_loss": 1.5})

        def evaluate(self):
            return {"eval_loss": 1.0}

        def save_state(self):
            return None

    trl.SFTConfig = _sft_config if sft_config_class is None else sft_config_class
    trl.SFTTrainer = _SFTTrainer
    sys.modules["trl"] = trl
    return saved


def _restore_ml_stubs(saved: dict) -> None:
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _write_split(tmp: Path, train_count: int = 48, validation_count: int = 6) -> tuple[Path, Path]:
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
            for i in range(validation_count)
        ),
        encoding="utf-8",
    )
    return train_file, val_file


_UNSET = object()

#: Every argument run_training() sends to SFTConfig, minus the two that were
#: renamed between the Tillicum pins and the VM's releases.
_SFT_COMMON_FIELDS = (
    "output_dir", "num_train_epochs", "per_device_train_batch_size",
    "per_device_eval_batch_size", "gradient_accumulation_steps", "learning_rate",
    "weight_decay", "logging_steps", "eval_strategy", "save_strategy",
    "save_total_limit", "bf16", "fp16", "use_cpu", "dataloader_pin_memory",
    "gradient_checkpointing", "report_to", "seed", "packing",
    "dataset_text_field", "max_steps", "eval_steps", "save_steps",
)
#: transformers 4.47.1 / TRL 0.13 (the cluster).
_SFT_OLD_FIELDS = _SFT_COMMON_FIELDS + ("max_seq_length", "warmup_ratio")
#: transformers 5.16.1 / TRL 1.12.0 (the VM's cpu-training-venv).
_SFT_CURRENT_FIELDS = _SFT_COMMON_FIELDS + ("max_length", "warmup_steps")


def _strict_sft_config_class(captured: dict, field_names: tuple[str, ...]):
    """A dataclass with exactly these init parameters, like the real SFTConfig.

    Records the arguments it was constructed with (not its defaults) so a test
    sees what reached the constructor. An unknown name is a TypeError, as for
    any dataclass, and inspect.signature() lists exactly these names.
    """

    def __post_init__(self):
        captured["sft"] = {
            name: getattr(self, name)
            for name in field_names
            if getattr(self, name) is not _UNSET
        }

    return dataclasses.make_dataclass(
        "SFTConfig",
        [(name, object, dataclasses.field(default=_UNSET)) for name in field_names],
        namespace={"__post_init__": __post_init__},
    )


class DeviceSelectionTests(unittest.TestCase):
    def test_default_is_cuda_and_cpu_is_explicit(self) -> None:
        args = train_qlora.parse_args([])
        self.assertFalse(args.cpu)
        self.assertIsNone(args.cpu_threads)
        self.assertEqual(train_qlora.resolve_device(args.cpu), "cuda")

        cpu_args = train_qlora.parse_args(["--cpu", "--cpu-threads", "6"])
        self.assertTrue(cpu_args.cpu)
        self.assertEqual(cpu_args.cpu_threads, 6)
        self.assertEqual(train_qlora.resolve_device(cpu_args.cpu), "cpu")

    def test_cpu_defaults_keep_the_production_hyperparameters(self) -> None:
        args = train_qlora.parse_args(["--cpu"])
        self.assertEqual(args.lora_r, 8)
        self.assertEqual(args.lora_alpha, 16)
        self.assertEqual(args.lora_dropout, 0.05)
        self.assertEqual(args.max_seq_length, 512)
        self.assertEqual(args.learning_rate, 2e-4)
        self.assertEqual(args.epochs, 3.0)
        self.assertEqual(args.per_device_batch_size, 1)
        self.assertEqual(args.gradient_accumulation_steps, 8)
        self.assertEqual(args.warmup_ratio, 0.1)
        self.assertEqual(args.weight_decay, 0.01)
        self.assertEqual(args.seed, 360)
        self.assertEqual(args.model_id, "meta-llama/Llama-3.2-3B-Instruct")

    def test_option_mixes_are_refused_not_guessed(self) -> None:
        with self.assertRaises(ValueError) as gpu_count:
            train_qlora.validate_device_arguments(
                train_qlora.parse_args(["--cpu", "--gpu-count", "1"])
            )
        self.assertIn("--gpu-count", str(gpu_count.exception))

        with self.assertRaises(ValueError) as batch:
            train_qlora.validate_device_arguments(
                train_qlora.parse_args(["--cpu", "--per-device-batch-size", "2"])
            )
        self.assertIn("batch-size 1", str(batch.exception))

        with self.assertRaises(ValueError):
            train_qlora.validate_device_arguments(
                train_qlora.parse_args(["--cpu", "--cpu-threads", "0"])
            )

        with self.assertRaises(ValueError) as threads:
            train_qlora.validate_device_arguments(
                train_qlora.parse_args(["--cpu-threads", "4"])
            )
        self.assertIn("--cpu", str(threads.exception))

        # The CUDA path accepts what it always accepted.
        train_qlora.validate_device_arguments(train_qlora.parse_args(["--gpu-count", "2"]))
        train_qlora.validate_device_arguments(train_qlora.parse_args([]))

    def test_main_reports_the_mix_and_exits_nonzero_before_importing_anything(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = train_qlora.main(["--cpu", "--gpu-count", "1"])
        self.assertEqual(status, 1)
        self.assertIn("mutually exclusive", stderr.getvalue())


class PureSettingsTests(unittest.TestCase):
    def test_quantization_is_nf4_double_quant_in_both_modes(self) -> None:
        cpu = train_qlora.quantization_settings("float32")
        cuda = train_qlora.quantization_settings("bfloat16")
        for settings in (cpu, cuda):
            self.assertTrue(settings["load_in_4bit"])
            self.assertEqual(settings["bnb_4bit_quant_type"], "nf4")
            self.assertTrue(settings["bnb_4bit_use_double_quant"])
        self.assertEqual(cpu["bnb_4bit_compute_dtype"], "float32")
        self.assertEqual(cuda["bnb_4bit_compute_dtype"], "bfloat16")

    def test_cpu_model_placement_is_a_dict_of_cpu_and_cuda_is_auto(self) -> None:
        self.assertEqual(
            train_qlora.model_load_settings("cpu", "float32"),
            {"device_map": {"": "cpu"}, "torch_dtype": "float32"},
        )
        self.assertEqual(
            train_qlora.model_load_settings("cuda", "bfloat16"),
            {"device_map": "auto", "torch_dtype": "bfloat16"},
        )
        with self.assertRaises(ValueError):
            train_qlora.model_load_settings("mps", "float32")

    def test_cpu_precision_has_no_autocast_and_cuda_keeps_bf16_or_fp16(self) -> None:
        cpu = train_qlora.precision_settings("cpu", False)
        self.assertEqual(
            cpu,
            {"bf16": False, "fp16": False, "use_cpu": True, "dataloader_pin_memory": False},
        )
        with self.assertRaises(ValueError):
            train_qlora.precision_settings("cpu", True)
        self.assertEqual(train_qlora.precision_settings("cuda", True), {"bf16": True, "fp16": False})
        self.assertEqual(train_qlora.precision_settings("cuda", False), {"bf16": False, "fp16": True})

    def test_dtype_names(self) -> None:
        self.assertEqual(train_qlora.dtype_name("torch.float32"), "float32")
        self.assertEqual(train_qlora.dtype_name("bfloat16"), "bfloat16")

    def test_chat_formatting_is_the_model_chat_template_of_a_user_assistant_pair(self) -> None:
        calls = []

        class _Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                calls.append((messages, kwargs))
                return "formatted"

        result = train_qlora.format_chat_example(
            _Tokenizer(), {"instruction": "When is the final?", "response": "Week 11."}
        )
        self.assertEqual(result, {"text": "formatted"})
        messages, kwargs = calls[0]
        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "When is the final?"},
                {"role": "assistant", "content": "Week 11."},
            ],
        )
        self.assertEqual(kwargs, {"tokenize": False, "add_generation_prompt": False})

    def test_peak_rss_is_a_positive_byte_count(self) -> None:
        peak = train_qlora.peak_rss_bytes()
        self.assertIsInstance(peak, int)
        assert peak is not None
        # Any Python process is well above 1 MiB and below 1 TiB.
        self.assertGreater(peak, 1024 * 1024)
        self.assertLess(peak, 1024 ** 4)

    def test_library_versions_never_raise_for_missing_packages(self) -> None:
        versions = train_qlora.collect_library_versions(("json", "no_such_package_css360"))
        self.assertIn("json", versions)
        self.assertIsNone(versions["no_such_package_css360"])


class SftConfigCompatibilityTests(unittest.TestCase):
    """build_sft_config() across the cluster pins and the VM's newer releases.

    The first real CPU smoke run failed with ``SFTConfig.__init__() got an
    unexpected keyword argument 'warmup_ratio'`` on transformers 5.16.1 / TRL
    1.12.0, after the earlier retry had only anticipated ``max_seq_length``.
    """

    def setUp(self) -> None:
        self.captured: dict = {}
        self.kwargs = {
            "max_seq_length": 512,
            "warmup_ratio": 0.1,
            "packing": False,
            "learning_rate": 2e-4,
        }

    def test_warmup_steps_from_ratio_is_the_old_scheduler_rounding(self) -> None:
        cases = [
            # (ratio, optimizer steps, expected): ceil(steps * ratio)
            (0.1, 3, 1),     # smoke run
            (0.1, 18, 2),    # CSS 360: 48 examples / batch 8 / 3 epochs
            (0.1, 15, 2),    # CSS 350: 37 examples / batch 8 / 3 epochs
            (0.1, 1, 1),
            (0.0, 18, 0),
            (1.0, 18, 18),
        ]
        for ratio, steps, expected in cases:
            with self.subTest(ratio=ratio, steps=steps):
                self.assertEqual(train_qlora.warmup_steps_from_ratio(ratio, steps), expected)
        with self.assertRaises(ValueError):
            train_qlora.warmup_steps_from_ratio(1.5, 18)
        with self.assertRaises(ValueError):
            train_qlora.warmup_steps_from_ratio(0.1, 0)

    def test_the_cluster_signature_receives_the_pinned_names_unchanged(self) -> None:
        old_trl = _strict_sft_config_class(
            self.captured, ("max_seq_length", "warmup_ratio", "packing", "learning_rate")
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            config = train_qlora.build_sft_config(
                old_trl, self.kwargs, total_optimizer_steps=18
            )
        self.assertEqual(self.captured["sft"], self.kwargs)
        self.assertEqual(config.warmup_ratio, 0.1)
        self.assertEqual(config.max_seq_length, 512)
        self.assertNotIn("SFTConfig compatibility", stdout.getvalue())

    def test_the_current_signature_gets_max_length_and_ceil_warmup_steps(self) -> None:
        new_trl = _strict_sft_config_class(
            self.captured, ("max_length", "warmup_steps", "packing", "learning_rate")
        )
        for steps, expected_warmup in ((3, 1), (18, 2), (15, 2)):
            with self.subTest(steps=steps):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    train_qlora.build_sft_config(
                        new_trl, self.kwargs, total_optimizer_steps=steps
                    )
                self.assertEqual(
                    self.captured["sft"],
                    {
                        "max_length": 512,
                        "warmup_steps": expected_warmup,
                        "packing": False,
                        "learning_rate": 2e-4,
                    },
                )
                self.assertIn("max_seq_length -> max_length=512", stdout.getvalue())
                self.assertIn(
                    f"warmup_ratio=0.1 -> warmup_steps={expected_warmup} of {steps}",
                    stdout.getvalue(),
                )
        # The caller's dict is not mutated.
        self.assertEqual(self.kwargs["max_seq_length"], 512)
        self.assertEqual(self.kwargs["warmup_ratio"], 0.1)

    def test_an_unsupported_argument_is_an_error_naming_it_not_a_silent_drop(self) -> None:
        no_packing = _strict_sft_config_class(
            self.captured, ("max_length", "warmup_steps", "learning_rate")
        )
        with self.assertRaises(ValueError) as ctx:
            train_qlora.build_sft_config(no_packing, self.kwargs, total_optimizer_steps=18)
        self.assertIn("packing", str(ctx.exception))
        self.assertIn("does not accept", str(ctx.exception))
        self.assertNotIn("sft", self.captured)

    def test_a_release_with_neither_warmup_name_is_an_error(self) -> None:
        no_warmup = _strict_sft_config_class(
            self.captured, ("max_length", "packing", "learning_rate")
        )
        with self.assertRaises(ValueError) as ctx:
            train_qlora.build_sft_config(no_warmup, self.kwargs, total_optimizer_steps=18)
        self.assertIn("warmup_ratio", str(ctx.exception))
        self.assertNotIn("sft", self.captured)

    def test_a_release_with_neither_length_name_is_an_error(self) -> None:
        no_length = _strict_sft_config_class(
            self.captured, ("warmup_steps", "packing", "learning_rate")
        )
        with self.assertRaises(ValueError) as ctx:
            train_qlora.build_sft_config(no_length, self.kwargs, total_optimizer_steps=18)
        self.assertIn("max_seq_length", str(ctx.exception))

    def test_a_var_keyword_stand_in_passes_everything_through(self) -> None:
        seen = {}

        def stand_in(**kwargs):
            seen.update(kwargs)
            return types.SimpleNamespace(**kwargs)

        train_qlora.build_sft_config(stand_in, self.kwargs, total_optimizer_steps=18)
        self.assertEqual(seen, self.kwargs)
        self.assertIsNone(train_qlora.supported_config_parameters(stand_in))
        self.assertEqual(
            train_qlora.supported_config_parameters(
                _strict_sft_config_class({}, ("max_length", "warmup_steps"))
            ),
            {"max_length", "warmup_steps"},
        )


class CpuRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.captured: dict = {}

    def _run(self, argv_extra: list[str], *, cuda_available: bool = False,
             bf16_supported: bool = False, train_count: int = 48, smoke: bool = True,
             sft_config_class=None):
        saved = _install_ml_stubs(
            self.captured, cuda_available=cuda_available, bf16_supported=bf16_supported,
            sft_config_class=sft_config_class,
        )
        self.addCleanup(_restore_ml_stubs, saved)
        tmp = Path(tempfile.mkdtemp())
        train_file, val_file = _write_split(tmp, train_count=train_count)
        argv = [
            "--train-file", str(train_file),
            "--validation-file", str(val_file),
            "--output-dir", str(tmp / "out"),
            *argv_extra,
        ]
        if smoke:
            argv.append("--smoke-test")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            report = train_qlora.run_training(train_qlora.parse_args(argv))
        return report, tmp / "out", stdout.getvalue()

    def test_cpu_smoke_builds_the_fp32_cpu_configuration(self) -> None:
        def _no_cuda_gate():
            raise AssertionError("require_cuda must not run in CPU mode")

        original = train_qlora.require_cuda
        train_qlora.require_cuda = _no_cuda_gate
        self.addCleanup(setattr, train_qlora, "require_cuda", original)

        report, out_dir, stdout = self._run(["--cpu", "--cpu-threads", "6"])

        bnb = self.captured["bnb"]
        self.assertEqual(bnb["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(bnb["bnb_4bit_use_double_quant"])
        self.assertEqual(bnb["bnb_4bit_compute_dtype"], "float32")
        self.assertTrue(bnb["load_in_4bit"])

        load = self.captured["from_pretrained"]
        self.assertEqual(load["model_id"], "meta-llama/Llama-3.2-3B-Instruct")
        self.assertEqual(load["device_map"], {"": "cpu"})
        self.assertEqual(load["torch_dtype"], "float32")
        self.assertIs(load["quantization_config"], bnb)

        lora = self.captured["lora"]
        self.assertEqual(lora["r"], 8)
        self.assertEqual(lora["lora_alpha"], 16)
        self.assertEqual(lora["lora_dropout"], 0.05)
        self.assertEqual(lora["bias"], "none")
        self.assertEqual(lora["task_type"], "CAUSAL_LM")
        self.assertEqual(
            lora["target_modules"],
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.assertTrue(self.captured.get("gradient_checkpointing_enabled"))

        sft = self.captured["sft"]
        self.assertFalse(sft["bf16"])
        self.assertFalse(sft["fp16"])
        self.assertTrue(sft["use_cpu"])
        self.assertFalse(sft["dataloader_pin_memory"])
        self.assertEqual(sft["per_device_train_batch_size"], 1)
        self.assertEqual(sft["per_device_eval_batch_size"], 1)
        self.assertEqual(sft["gradient_accumulation_steps"], 8)
        self.assertEqual(sft["learning_rate"], 2e-4)
        self.assertEqual(sft["num_train_epochs"], 3.0)
        self.assertEqual(sft["warmup_ratio"], 0.1)
        self.assertEqual(sft["weight_decay"], 0.01)
        self.assertEqual(sft["seed"], 360)
        self.assertEqual(sft["max_seq_length"], 512)
        self.assertFalse(sft["packing"])
        self.assertTrue(sft["gradient_checkpointing"])
        self.assertEqual(sft["max_steps"], 3)

        self.assertEqual(self.captured["set_num_threads"], 6)
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["cpuThreads"], 6)
        self.assertEqual(report["computeDtype"], "float32")
        self.assertEqual(report["gpuCount"], 0)
        self.assertIsNone(report["estimatedGpuHours"])
        self.assertIsNone(report["actualGpuHours"])
        self.assertIn("CPU mode: 6 torch threads, float32 compute", stdout)
        self.assertIn("Device: cpu (6 torch threads, float32 compute)", stdout)

    def test_cpu_mode_leaves_an_available_cuda_device_alone(self) -> None:
        report, _, stdout = self._run(["--cpu"], cuda_available=True, bf16_supported=True)
        self.assertEqual(self.captured["from_pretrained"]["device_map"], {"": "cpu"})
        self.assertEqual(self.captured["bnb"]["bnb_4bit_compute_dtype"], "float32")
        self.assertFalse(self.captured["sft"]["bf16"])
        self.assertFalse(self.captured["sft"]["fp16"])
        self.assertTrue(self.captured["sft"]["use_cpu"])
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["gpuCount"], 0)
        self.assertIn("CUDA device will not be used", stdout)
        # No thread override asked for: torch's default is reported, not changed.
        self.assertNotIn("set_num_threads", self.captured)
        self.assertEqual(report["cpuThreads"], 8)

    def test_cpu_full_run_uses_the_same_step_budget_as_the_cluster(self) -> None:
        report, out_dir, _ = self._run(["--cpu"], smoke=False, train_count=48)
        sft = self.captured["sft"]
        self.assertEqual(sft["max_steps"], 18)
        self.assertEqual(sft["eval_strategy"], "epoch")
        self.assertEqual(sft["save_strategy"], "epoch")
        self.assertEqual(report["effectiveBatchSize"], 8)
        self.assertEqual(report["intendedOptimizerSteps"], 18)
        self.assertEqual(report["completedSteps"], 18)
        self.assertTrue(report["trainingLengthSatisfied"])
        self.assertIsNone(report["actualGpuHours"])
        self.assertEqual(report["gpuCount"], 0)

    def test_cuda_configuration_is_unchanged(self) -> None:
        report, out_dir, _ = self._run([], cuda_available=True, bf16_supported=True)

        bnb = self.captured["bnb"]
        self.assertEqual(
            bnb,
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "bfloat16",
            },
        )
        load = self.captured["from_pretrained"]
        self.assertEqual(load["device_map"], "auto")
        self.assertEqual(load["torch_dtype"], "bfloat16")

        sft = self.captured["sft"]
        self.assertTrue(sft["bf16"])
        self.assertFalse(sft["fp16"])
        self.assertNotIn("use_cpu", sft)
        self.assertNotIn("dataloader_pin_memory", sft)
        self.assertEqual(sft["max_seq_length"], 512)
        self.assertTrue(sft["gradient_checkpointing"])
        self.assertNotIn("set_num_threads", self.captured)

        self.assertEqual(report["device"], "cuda")
        self.assertEqual(report["computeDtype"], "bfloat16")
        self.assertIsNone(report["cpuThreads"])
        self.assertEqual(report["gpuCount"], 1)
        self.assertIsNotNone(report["estimatedGpuHours"])

        resolved = json.loads((out_dir / "resolved_config.json").read_text(encoding="utf-8"))
        self.assertEqual(resolved["device"], "cuda")
        self.assertEqual(resolved["compute_dtype"], "bfloat16")
        self.assertEqual(resolved["gpu_count"], 1)
        self.assertIsNone(resolved["cpu_threads"])

    def test_cuda_without_bf16_still_falls_to_fp16_as_before(self) -> None:
        report, _, _ = self._run([], cuda_available=True, bf16_supported=False)
        self.assertEqual(self.captured["bnb"]["bnb_4bit_compute_dtype"], "float16")
        self.assertEqual(self.captured["from_pretrained"]["torch_dtype"], "float16")
        self.assertFalse(self.captured["sft"]["bf16"])
        self.assertTrue(self.captured["sft"]["fp16"])
        self.assertEqual(report["computeDtype"], "float16")

    def test_cuda_mode_still_requires_cuda(self) -> None:
        saved = _install_ml_stubs(self.captured, cuda_available=False, bf16_supported=False)
        self.addCleanup(_restore_ml_stubs, saved)
        with self.assertRaises(RuntimeError) as ctx:
            train_qlora.run_training(train_qlora.parse_args(["--smoke-test"]))
        self.assertIn("CUDA is unavailable", str(ctx.exception))

    def test_both_modes_train_on_identical_chat_text(self) -> None:
        _, _, _ = self._run(["--cpu"])
        cpu_train = list(self.captured["train_texts"])
        cpu_eval = list(self.captured["eval_texts"])
        cpu_calls = list(self.captured["chat_template_calls"])
        self.captured.clear()

        _, _, _ = self._run([], cuda_available=True, bf16_supported=True)
        self.assertEqual(self.captured["train_texts"], cpu_train)
        self.assertEqual(self.captured["eval_texts"], cpu_eval)
        self.assertEqual(self.captured["chat_template_calls"], cpu_calls)

        # Smoke subset of the same split: 4 train / 2 validation, user then assistant.
        self.assertEqual(len(cpu_train), 4)
        self.assertEqual(len(cpu_eval), 2)
        first = cpu_calls[0]
        self.assertEqual([m["role"] for m in first["messages"]], ["user", "assistant"])
        self.assertEqual(first["messages"][0]["content"], "Q0?")
        self.assertEqual(first["messages"][1]["content"], "A0.")
        self.assertEqual(first["kwargs"], {"tokenize": False, "add_generation_prompt": False})
        self.assertIn(json.dumps(first["messages"]), cpu_train[0])

    def test_cpu_output_metadata_and_artifacts(self) -> None:
        report, out_dir, _ = self._run(["--cpu", "--cpu-threads", "4"])

        resolved = json.loads((out_dir / "resolved_config.json").read_text(encoding="utf-8"))
        self.assertEqual(resolved["mode"], "smoke")
        self.assertEqual(resolved["device"], "cpu")
        self.assertEqual(resolved["compute_dtype"], "float32")
        self.assertEqual(resolved["cpu_threads"], 4)
        self.assertEqual(resolved["gpu_count"], 0)
        self.assertEqual(resolved["per_device_train_batch_size"], 1)
        self.assertEqual(resolved["gradient_accumulation_steps"], 8)
        self.assertEqual(resolved["max_seq_length"], 512)
        self.assertEqual(resolved["lora_r"], 8)
        self.assertEqual(resolved["lora_alpha"], 16)
        self.assertEqual(resolved["lora_dropout"], 0.05)
        self.assertEqual(
            resolved["lora_target_modules"],
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.assertEqual(resolved["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(resolved["bnb_4bit_use_double_quant"])
        self.assertTrue(resolved["gradient_checkpointing"])
        self.assertEqual(resolved["model_id"], "meta-llama/Llama-3.2-3B-Instruct")
        self.assertEqual(resolved["full_train_example_count"], 48)
        self.assertEqual(resolved["full_validation_example_count"], 6)

        written = json.loads((out_dir / "runtime-report.json").read_text(encoding="utf-8"))
        self.assertEqual(written["device"], "cpu")
        self.assertEqual(written["computeDtype"], "float32")
        self.assertEqual(written["cpuThreads"], 4)
        self.assertEqual(written["gpuCount"], 0)
        self.assertEqual(written["modelId"], "meta-llama/Llama-3.2-3B-Instruct")
        self.assertIsNone(written["estimatedGpuHours"])
        self.assertIsNone(written["actualGpuHours"])
        self.assertIsNone(written["slurmJobId"])
        for key in ("modelLoadSeconds", "trainingSeconds", "evaluationSeconds",
                    "totalElapsedSeconds", "averageSecondsPerStep"):
            self.assertIsInstance(written[key], (int, float), key)
        self.assertIsInstance(written["peakRssBytes"], int)
        self.assertGreater(written["peakRssBytes"], 0)
        self.assertAlmostEqual(written["peakRssMiB"], written["peakRssBytes"] / (1024 * 1024))
        self.assertIsInstance(written["peakRssAfterModelLoadBytes"], int)
        self.assertLessEqual(written["peakRssAfterModelLoadBytes"], written["peakRssBytes"])
        self.assertEqual(written["libraryVersions"]["torch"], "0.0-stub")
        self.assertIn("transformers", written["libraryVersions"])
        # The smoke run still projects the full run from the full split.
        self.assertEqual(written["estimatedOptimizerSteps"], 18)
        self.assertIsNotNone(written["estimatedTrainingOnlySeconds"])
        self.assertIsNotNone(written["estimatedConservativeTotalSeconds"])
        self.assertEqual(report["peakRssBytes"], written["peakRssBytes"])

        # Standard PEFT adapter layout, where report_training_result.py and the
        # GGUF conversion look for it.
        self.assertTrue((out_dir / "adapter" / "adapter_config.json").is_file())
        self.assertTrue((out_dir / "adapter" / "adapter_model.safetensors").is_file())
        self.assertTrue((out_dir / "tokenizer" / "tokenizer_config.json").is_file())
        self.assertTrue((out_dir / "training_metrics.json").is_file())
        self.assertTrue((out_dir / "evaluation_metrics.json").is_file())

    def test_cpu_smoke_on_the_current_releases_gets_one_warmup_step(self) -> None:
        current = _strict_sft_config_class(self.captured, _SFT_CURRENT_FIELDS)
        report, _, stdout = self._run(["--cpu"], sft_config_class=current)
        sft = self.captured["sft"]
        self.assertEqual(sft["max_steps"], 3)
        self.assertEqual(sft["warmup_steps"], 1)
        self.assertEqual(sft["max_length"], 512)
        self.assertNotIn("warmup_ratio", sft)
        self.assertNotIn("max_seq_length", sft)
        self.assertEqual(sft["learning_rate"], 2e-4)
        self.assertTrue(sft["use_cpu"])
        self.assertFalse(sft["bf16"])
        self.assertFalse(sft["fp16"])
        self.assertIn("warmup_ratio=0.1 -> warmup_steps=1 of 3 optimizer steps", stdout)
        self.assertEqual(report["completedSteps"], 3)
        self.assertEqual(report["device"], "cpu")

    def test_cpu_full_css360_on_the_current_releases_gets_two_warmup_steps(self) -> None:
        current = _strict_sft_config_class(self.captured, _SFT_CURRENT_FIELDS)
        report, _, stdout = self._run(
            ["--cpu"], smoke=False, train_count=48, sft_config_class=current
        )
        sft = self.captured["sft"]
        self.assertEqual(sft["max_steps"], 18)
        self.assertEqual(sft["warmup_steps"], 2)
        self.assertEqual(sft["max_length"], 512)
        self.assertNotIn("warmup_ratio", sft)
        self.assertIn("warmup_ratio=0.1 -> warmup_steps=2 of 18 optimizer steps", stdout)
        self.assertEqual(report["intendedOptimizerSteps"], 18)
        self.assertTrue(report["trainingLengthSatisfied"])

    def test_cuda_on_the_cluster_releases_still_sends_warmup_ratio_and_max_seq_length(self) -> None:
        old = _strict_sft_config_class(self.captured, _SFT_OLD_FIELDS)
        report, _, stdout = self._run(
            [], cuda_available=True, bf16_supported=True, smoke=False, train_count=48,
            sft_config_class=old,
        )
        sft = self.captured["sft"]
        self.assertEqual(sft["warmup_ratio"], 0.1)
        self.assertEqual(sft["max_seq_length"], 512)
        self.assertNotIn("warmup_steps", sft)
        self.assertNotIn("max_length", sft)
        self.assertEqual(sft["max_steps"], 18)
        self.assertTrue(sft["bf16"])
        self.assertNotIn("SFTConfig compatibility", stdout)
        self.assertEqual(report["device"], "cuda")


if __name__ == "__main__":
    unittest.main()
