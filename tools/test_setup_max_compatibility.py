"""--max-compatibility: the Windows portable build (STRATA_PORTABLE) of both engines.

build_engine (the Qwen engine and its image encoder) and build_engine_poly (strata-poly and its encoder) pass
STRATA_PORTABLE=ON, record "portable" in BUILD.json, compile again when the setting changes, and the strata-poly
stamp no longer claims an image encoder built with the other setting.

    python -m unittest tools.test_setup_max_compatibility
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

GPU = {"arch": 86, "archs": [86], "name": "RTX 3060"}
POLY_GPU = {"arch": 89, "archs": [89], "driver": "580.0"}


def binary(version: str) -> bytes:
    return b"\x7fELF....tail_role_token=0 vram_elastic=0 engine=" + version.encode() + b"\n...."


class BuildEnginePortable(unittest.TestCase):
    """build_engine (CUDA): the Qwen engine and its encoder."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "engine").mkdir()
        self.compiles = 0
        self.made = {}
        self.floors = []
        self.warned = []

    def build(self, portable=False, vision="none", floor=""):
        self.made = {}

        def cmake(src, bdir, target, defs, vcvars, script):
            self.compiles += 1
            self.made[target] = list(defs)
            out = bdir / "bin" if target == "strata-vision" else bdir
            out.mkdir(parents=True, exist_ok=True)
            (out / (setup.VEXE if target == "strata-vision" else setup.EXE)).write_bytes(binary("0.1.45"))

        def isa_floor_defs(value, bdir, meta):
            self.floors.append(value)
            return []

        patches = [mock.patch.object(setup, "ROOT", self.root),
                   mock.patch.object(setup, "source_hash", lambda parts: "newhash"),
                   mock.patch.object(setup, "source_version", lambda: "0.1.45"),
                   mock.patch.object(setup, "install_build_tools", lambda gpu, yes: ("nvcc", None)),
                   mock.patch.object(setup, "toolkit_root_defs", lambda nvcc: []),
                   mock.patch.object(setup, "host_compiler_defs", lambda cuda: []),
                   mock.patch.object(setup, "engine_defs", lambda archs, toolkit=13: []),
                   mock.patch.object(setup, "isa_floor_defs", isa_floor_defs),
                   mock.patch.object(setup, "cpu_floor", lambda avx2: floor),
                   mock.patch.object(setup, "cpu_info", lambda: ("cpu", True)),
                   mock.patch.object(setup, "cmake_build", cmake),
                   mock.patch.object(setup, "warn", lambda message, *a, **k: self.warned.append(message)),
                   mock.patch.object(setup, "say", lambda *a, **k: None),
                   mock.patch.object(setup, "ok", lambda *a, **k: None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        setup.build_engine(GPU, vision, True, Path("llama"), toolkit=13, portable=portable)
        return self.made

    def stamp(self):
        return json.loads((self.root / "engine" / "BUILD.json").read_text(encoding="utf-8"))

    def test_portable_is_stamped_kept_and_rebuilt_when_it_changes(self):
        made = self.build(portable=True)
        self.assertIn("-DSTRATA_PORTABLE=ON", made["strata"])
        self.assertTrue(self.stamp()["portable"])
        made = self.build(portable=True)
        self.assertEqual((self.compiles, made), (1, {}))        # this one is kept
        made = self.build()
        self.assertIn("-DSTRATA_PORTABLE=OFF", made["strata"])
        self.assertEqual(self.compiles, 2)                      # the setting changed: built again
        self.assertNotIn("portable", self.stamp())

    def test_portable_clears_an_older_cpu_floor(self):
        self.build(portable=True, floor="avx")
        self.assertEqual(self.floors, [""])                     # CMake refuses both: the portable build wins
        self.assertTrue(any("--max-compatibility" in m and "STRATA_ISA_FLOOR=avx" in m for m in self.warned), self.warned)
        self.assertNotIn("isa_floor", self.stamp())
        self.assertIn("-DSTRATA_PORTABLE=ON", self.made["strata"])

    def test_the_image_encoder_is_portable_too(self):
        made = self.build(portable=True, vision="cpu")
        self.assertIn("-DSTRATA_PORTABLE=ON", made["strata"])
        self.assertIn("-DSTRATA_PORTABLE=ON", made["strata-vision"])
        self.assertEqual(self.stamp()["vision"], "cpu")
        made = self.build(portable=False, vision="cpu")
        self.assertIn("-DSTRATA_PORTABLE=OFF", made["strata-vision"])   # the encoder of the other setting is replaced
        self.assertEqual(self.compiles, 4)                              # the engine and the encoder, twice


class BuildEnginePolyPortable(unittest.TestCase):
    """build_engine_poly (CUDA): strata-poly and its encoder."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / setup.ENGINE_POLY_DIR).mkdir()
        self.compiles = 0
        self.made = {}

    def build(self, portable=False, vision="none"):
        self.made = {}

        def cmake(src, bdir, target, defs, vcvars, script):
            self.compiles += 1
            self.made[target] = list(defs)
            out = bdir / "bin" if target == "strata-vision" else bdir
            out.mkdir(parents=True, exist_ok=True)
            (out / (setup.VEXE if target == "strata-vision" else setup.POLY_EXE)).write_text("x")

        patches = [mock.patch.object(setup, "ROOT", self.root),
                   mock.patch.object(setup, "source_hash", lambda parts: "newhash"),
                   mock.patch.object(setup, "source_version", lambda: "0.1.45"),
                   mock.patch.object(setup, "install_build_tools", lambda gpu, yes: ("nvcc", None)),
                   mock.patch.object(setup, "cmake_build", cmake),
                   mock.patch.object(setup, "cuda_lib_dirs", lambda toolkit: []),
                   mock.patch.object(setup, "get_llama_cpp", lambda vision=False: Path("llama")),
                   mock.patch.object(setup, "say", lambda *a, **k: None),
                   mock.patch.object(setup, "ok", lambda *a, **k: None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        setup.build_engine_poly(POLY_GPU, True, Path("llama"), vision, 13, portable)
        return self.made

    def stamp(self):
        return json.loads((self.root / setup.ENGINE_POLY_DIR / "BUILD.json").read_text(encoding="utf-8"))

    def test_portable_is_stamped_kept_and_rebuilt_when_it_changes(self):
        made = self.build(portable=True)
        self.assertIn("-DSTRATA_PORTABLE=ON", made["strata-poly"])
        self.assertTrue(self.stamp()["portable"])
        made = self.build(portable=True)
        self.assertEqual((self.compiles, made), (1, {}))
        made = self.build()
        self.assertIn("-DSTRATA_PORTABLE=OFF", made["strata-poly"])
        self.assertNotIn("portable", self.stamp())

    def test_the_image_encoder_is_portable_and_rebuilt_when_it_changes(self):
        made = self.build(portable=True, vision="gpu")
        self.assertIn("-DSTRATA_PORTABLE=ON", made["strata-poly"])
        self.assertIn("-DSTRATA_PORTABLE=ON", made["strata-vision"])
        made = self.build(portable=False, vision="gpu")
        self.assertIn("-DSTRATA_PORTABLE=OFF", made["strata-vision"])
        made = self.build(portable=True, vision="gpu")
        self.assertIn("-DSTRATA_PORTABLE=ON", made["strata-vision"])

    def test_a_vision_none_run_does_not_claim_the_old_encoder(self):
        self.build(portable=False, vision="gpu")
        self.assertEqual(self.stamp()["vision"], "gpu")
        self.build(portable=True, vision="none")
        meta = self.stamp()
        self.assertEqual((meta["vision"], meta["vision_src"]), ("none", None))
        self.assertTrue(meta["portable"])
        made = self.build(portable=True, vision="gpu")
        self.assertIn("strata-vision", made)                    # the old encoder was not claimed: built again


if __name__ == "__main__":
    unittest.main()
