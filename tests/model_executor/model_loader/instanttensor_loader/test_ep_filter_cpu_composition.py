"""CPU execution of composed production selection; no native/GPU claim."""

import ast
import contextlib
from pathlib import Path
import os
import re
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[4]
WU = ROOT / "vllm/model_executor/model_loader/weight_utils.py"


def function(source, name, namespace):
    tree = ast.parse(source.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
    return namespace[name]


class Metadata:
    def __init__(self, names):
        self.names = names

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def keys(self):
        return self.names

    def offset_keys(self):
        return self.names


class CompositionTest(unittest.TestCase):
    def setUp(self):
        self.ns = {"os": os, "re": re}
        ep = (ROOT / "vllm/model_executor/model_loader/ep_weight_filter.py").read_text()
        exec(ep.replace("import regex as re", "import re"), self.ns)

    def test_uniform_group_and_full_partition(self):
        calls = []
        groups = []
        names = [
            f"model.layers.0.mlp.experts.{i}.gate_proj.weight_packed"
            for i in range(512)
        ]
        scales = [
            f"model.layers.0.mlp.experts.{i}.gate_proj.weight_scale" for i in range(512)
        ]
        shared = [
            "model.layers[0].attn.weight",
            "mtp.layers.0.embed_tokens.weight",
            "model.layers.0.mlp.shared_experts.gate_proj.weight",
        ]
        keys = names + scales + shared

        @contextlib.contextmanager
        def group(g):
            groups.append(g)
            yield "existing-collective"

        sys.modules["vllm.model_executor.model_loader.instanttensor_utils"] = (
            types.SimpleNamespace(instanttensor_loading_group=group)
        )
        self.ns.update(
            current_platform=types.SimpleNamespace(is_cuda=lambda: True),
            get_world_group=lambda: "world",
            safe_open=lambda path, **kwargs: Metadata(keys),
            _instanttensor_weights_iterator=lambda *args, **kwargs: (
                calls.append(kwargs) or iter([])
            ),
        )
        run = function(WU, "instanttensor_weights_iterator", self.ns)
        for rank in range(16):
            ids = self.ns["compute_local_expert_ids"](512, 16, rank)
            list(run(["file"], False, local_expert_ids=ids))
            call = calls[-1]
            self.assertIsNone(call["process_group"])
            self.assertEqual(
                set(call["ep_filter_names"]) & set(names), {names[i] for i in ids}
            )
            self.assertTrue(set(scales + shared).issubset(call["ep_filter_names"]))
        self.assertEqual(groups, [])
        list(run(["file"], False, local_expert_ids=set()))
        self.assertEqual(set(calls[-1]["ep_filter_names"]), set(scales + shared))
        keys[:] = ["model.layers.0.mlp.experts.gate_up_proj.weight"]
        list(run(["file"], False, local_expert_ids={0}))
        self.assertEqual(calls[-1]["process_group"], "existing-collective")
        list(run(["file"], False, local_expert_ids=None))
        self.assertEqual(calls[-1]["process_group"], "existing-collective")
        self.assertEqual(groups, ["world", "world"])

    def test_public_filter_copy_and_generator_exit(self):
        calls = []
        events = []

        class Tensor:
            def numel(self):
                return 4

            def element_size(self):
                return 1

        class Native:
            ordered_tensor_metadatas = [("local", {})]
            total_tensor_size = 4

            def __enter__(self):
                events.append("open")
                return self

            def __exit__(self, *args):
                events.append("closed")

            def tensors(self):
                yield "local", Tensor()

        class Bar:
            def __init__(self, **kw):
                pass

            def update(self, *args):
                pass

            def close(self):
                events.append("bar-closed")

            _get_free_pos = staticmethod(lambda: 0)

        def opening(*args, **kw):
            calls.append(kw)
            return Native()

        sys.modules["instanttensor"] = types.SimpleNamespace(safe_open=opening)
        self.ns.update(
            current_platform=types.SimpleNamespace(
                is_cuda=lambda: True, current_device=lambda: 0
            ),
            tqdm=Bar,
            enable_tqdm=lambda value: False,
            _BAR_FORMAT="",
        )
        run = function(WU, "_instanttensor_weights_iterator", self.ns)
        old = os.environ.pop("INSTANTTENSOR_BUFFER_SIZE", None)
        try:
            generator = run(
                ["file"], False, ep_filter_names=["literal[0]"], process_group=None
            )
            name, tensor = next(generator)
            self.assertEqual(name, "local")
            generator.close()
            self.assertEqual(events, ["open", "bar-closed", "closed"])
            self.assertEqual(calls[0]["tensor_filter"], ["literal[0]"])
            self.assertEqual(calls[0]["tensor_name_match_mode"], "exact")
            self.assertIs(calls[0]["copy"], True)
            with self.assertRaisesRegex(RuntimeError, "independent"):
                list(
                    run(["file"], False, ep_filter_names=[], process_group="forbidden")
                )
        finally:
            if old is not None:
                os.environ["INSTANTTENSOR_BUFFER_SIZE"] = old

    def test_static_selector_new_abi_and_ep_mask(self):
        # Two physical files overlap a tensor name; index chooses overlay.
        physical = {
            "base": ["a", "dup", "skip", "big", "tail"],
            "overlay": ["dup", "extra"],
        }
        self.ns["safe_open"] = lambda path, **kwargs: Metadata(physical[path])
        run = function(WU, "_restrict_instanttensor_to_selected_ranges", self.ns)
        for filtered in [False, True]:
            all_names = physical["base"] + physical["overlay"]
            initial_keep = [True, True, not filtered, True, True, True, True]
            sizes = [4, 4, 4, 32, 4, 4, 4]
            metadata = [
                (n, {"data_offsets": [0, size]})
                for n, size, keep in zip(all_names, sizes, initial_keep)
                if keep
            ]
            finalized = []
            obj = types.SimpleNamespace(
                filename=["base", "overlay"],
                ordered_tensor_metadatas=metadata,
                tensor_offsets=[(0, v) for v in [100, 104, 108, 112, 144, 148]]
                + [(1, v) for v in [200, 204, 208]],
                tensor_keep=initial_keep,
                tensor_sizes=[],
                total_tensor_size=0,
                tensor_name_to_index={},
                loader_handle=None,
                _finalize_buffer_size=lambda value: finalized.append(value),
            )
            fallback = run(
                obj,
                indexed_tensor_files={
                    n: os.path.abspath("overlay" if n in ["dup", "extra"] else "base")
                    for n in set(all_names)
                },
                is_unused_weight=lambda name: name == "skip",
                max_tensor_size=16,
            )
            self.assertEqual(fallback, [("big", "base")])
            self.assertEqual(
                [n for n, _ in obj.ordered_tensor_metadatas],
                ["a", "tail", "dup", "extra"],
            )
            self.assertEqual(obj.tensor_keep, [True] * 4)
            self.assertEqual(obj.filename, ["base", "base", "overlay"])
            self.assertEqual(
                obj.tensor_offsets,
                [(0, 100), (0, 104), (1, 144), (1, 148), (2, 200), (2, 204), (2, 208)],
            )
            self.assertEqual(obj.total_tensor_size, 16)
            self.assertEqual(finalized, [16])
        obj.tensor_keep = [True]
        with self.assertRaisesRegex(RuntimeError, "count/type mismatch"):
            run(obj, indexed_tensor_files=None, is_unused_weight=None)


if __name__ == "__main__":
    unittest.main()
