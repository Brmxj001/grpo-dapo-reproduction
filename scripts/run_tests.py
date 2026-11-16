#!/usr/bin/env python3
"""统一测试入口：依次运行 tests/test_*.py，全部通过才返回 0。

    python scripts/run_tests.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"


def main() -> int:
    test_files = sorted(TESTS_DIR.glob("test_*.py"))
    if not test_files:
        print(f"没有找到测试文件：{TESTS_DIR}/test_*.py")
        return 1

    failed: list[str] = []
    for test_file in test_files:
        print("\n" + "=" * 70)
        print(f"运行 {test_file.relative_to(PROJECT_ROOT)}")
        print("=" * 70)
        result = subprocess.run([sys.executable, str(test_file)], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            failed.append(test_file.name)

    print("\n" + "=" * 70)
    if failed:
        print(f"测试失败：{', '.join(failed)}")
        return 1
    print(f"全部 {len(test_files)} 个测试文件通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
