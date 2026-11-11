"""通过真实进程验证题目输入输出，不依赖入口的内部实现。"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def invoke(self, *args):
        result = subprocess.run([sys.executable, str(ROOT / "main.py"), *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
        return json.loads(result.stdout)

    def test_full_original_demo(self):
        result = self.invoke()
        self.assertEqual(result["q1"][0], [0, 0])
        self.assertEqual(result["q1"][-1], [3, 4])
        self.assertEqual(len(result["q1"]) - 1, 7)
        self.assertEqual(result["q2"], 7)
        self.assertEqual(result["q3"], 5)
        self.assertEqual(result["q4"]["H=2 alphabeta"], {"best_target": "T2", "best_value": -1})
        self.assertEqual(result["q4"]["H=4 minimax"], {"best_target": "T3", "best_value": 3})

    def test_custom_json_inputs(self):
        cases = [
            ("q1", {"start": [0, 0], "goal": [0, 2], "grid": [["S", "X", "E"]]}, []),
            ("q2", {"start": [0, 0], "goal": [0, 1], "grid": [["S", "E"]]}, 0),
            ("q3", {"tree_structure": {"children": [{"value": -8}, {"value": -3}]}}, -3),
            ("q4", {"depth": 4, "order": "gain", "algorithm": "alphabeta"}, {"best_target": "T3", "best_value": 3}),
        ]
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "input.json"
            for question, data, expected in cases:
                path.write_text(json.dumps(data), encoding="utf-8")
                self.assertEqual(self.invoke(question, "--input", str(path)), expected)


if __name__ == "__main__":
    unittest.main()
