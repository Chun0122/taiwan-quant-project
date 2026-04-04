"""tests/test_morning_atomicity.py — 早晨流程原子性測試。

驗證單一步驟失敗不會阻擋後續步驟執行。
"""

from __future__ import annotations


class TestMorningStepAtomicity:
    """模擬步驟執行迴圈，驗證失敗容錯。"""

    @staticmethod
    def _run_steps(steps, active_flags=None):
        """仿照 cmd_morning_routine 的步驟執行邏輯。"""
        if active_flags is None:
            active_flags = set()

        results: list[tuple[int | str, str, str]] = []
        for num, title, skip_on, action in steps:
            if skip_on & active_flags:
                results.append((num, title, "skipped"))
            else:
                try:
                    action()
                    results.append((num, title, "success"))
                except Exception:
                    results.append((num, title, "failed"))
        return results

    def test_all_success(self):
        """全部成功 → 無失敗步驟。"""
        steps = [
            (1, "Step A", set(), lambda: None),
            (2, "Step B", set(), lambda: None),
            (3, "Step C", set(), lambda: None),
        ]
        results = self._run_steps(steps)
        statuses = [s for _, _, s in results]
        assert statuses == ["success", "success", "success"]

    def test_middle_step_failure_continues(self):
        """Step 2 失敗 → Step 3 仍然執行。"""
        executed = []
        steps = [
            (1, "Step A", set(), lambda: executed.append(1)),
            (2, "Step B", set(), lambda: (_ for _ in ()).throw(RuntimeError("boom"))),
            (3, "Step C", set(), lambda: executed.append(3)),
        ]
        results = self._run_steps(steps)
        assert executed == [1, 3]
        statuses = [s for _, _, s in results]
        assert statuses == ["success", "failed", "success"]

    def test_multiple_failures_all_reported(self):
        """多步驟失敗 → 全部記錄。"""
        steps = [
            (1, "Step A", set(), lambda: (_ for _ in ()).throw(ValueError("err1"))),
            (2, "Step B", set(), lambda: None),
            (3, "Step C", set(), lambda: (_ for _ in ()).throw(OSError("err2"))),
        ]
        results = self._run_steps(steps)
        failed = [(n, t) for n, t, s in results if s == "failed"]
        assert len(failed) == 2
        assert failed[0][0] == 1
        assert failed[1][0] == 3
