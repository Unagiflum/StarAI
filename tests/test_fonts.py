"""Text must measure the same on every platform: no font is looked up by name
from the operating system, and the monospaced font is bundled."""

import ast
import re
import unittest
from pathlib import Path

import pygame

import src.const as const
from src.UI import ui

pygame.font.init()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECS = ("StarAI.spec", "StarAI_CPUAI.spec", "StarAI_Train.spec", "StarAI_Linux.spec")
# SysFont("Name", ...) or Font("Name", ...) with a literal family name; None
# selects pygame's own bundled default font, which is the same everywhere.
_NAMED_SYSTEM_FONT = re.compile(r"""\b(?:SysFont|font\.Font)\(\s*["']""")


class BundledFontTests(unittest.TestCase):
    def test_the_monospaced_font_and_its_licence_are_bundled(self):
        self.assertTrue(const.MONOSPACE_FONT_PATH.is_file())
        licence = (const.MONOSPACE_FONT_PATH.parent / "LICENSE.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("Bitstream", licence)

    def test_the_monospaced_font_is_monospaced(self):
        font = ui.monospace_font(17)

        self.assertEqual(font.size("i")[0], font.size("M")[0])
        self.assertEqual(font.size("x" * 10)[0], 10 * font.size("x")[0])

    def test_every_build_packages_the_font_and_its_licence(self):
        for spec in SPECS:
            tree = ast.parse((PROJECT_ROOT / spec).read_text(encoding="utf-8"))
            extensions = next(
                ast.literal_eval(node.value)
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "asset_extensions"
                    for target in node.targets
                )
            )
            self.assertIn(".ttf", extensions, spec)
            self.assertIn(".txt", extensions, spec)

    def test_no_source_looks_up_an_operating_system_font_by_name(self):
        offenders = [
            f"{path.relative_to(PROJECT_ROOT)}:{number}"
            for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if _NAMED_SYSTEM_FONT.search(line)
        ]

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
