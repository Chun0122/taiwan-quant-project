"""P0 止血包 #7 — morning-routine 檔案鎖（_run_lock）測試。

flock 特性：同一 process 內對同一檔案的兩個獨立 fd 也互斥（LOCK_NB），
因此可在單一測試 process 內驗證互斥行為。
"""

from __future__ import annotations

import fcntl

import pytest

from src.cli.morning_cmd import _run_lock


class TestRunLock:
    def test_acquires_and_writes_pid(self, tmp_path):
        lock_path = tmp_path / "morning.lock"
        with _run_lock(lock_path):
            content = lock_path.read_text(encoding="utf-8")
            assert "pid=" in content
            assert "started=" in content

    def test_second_acquire_exits(self, tmp_path):
        """鎖被持有時，第二個 _run_lock 應 SystemExit(1)。"""
        lock_path = tmp_path / "morning.lock"
        # 模擬另一個執行中的 morning-routine：先以外部 fd 持鎖
        holder = open(lock_path, "w", encoding="utf-8")
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(SystemExit) as exc_info:
                with _run_lock(lock_path):
                    pytest.fail("不應取得鎖")
            assert exc_info.value.code == 1
        finally:
            holder.close()

    def test_lock_released_after_exit(self, tmp_path):
        """context 結束後鎖釋放，可再次取得。"""
        lock_path = tmp_path / "morning.lock"
        with _run_lock(lock_path):
            pass
        with _run_lock(lock_path):
            pass  # 第二次取得成功即證明釋放

    def test_lock_released_on_exception(self, tmp_path):
        """主體拋例外時鎖仍釋放（finally 保證）。"""
        lock_path = tmp_path / "morning.lock"
        with pytest.raises(RuntimeError):
            with _run_lock(lock_path):
                raise RuntimeError("boom")
        with _run_lock(lock_path):
            pass

    def test_creates_parent_dir(self, tmp_path):
        lock_path = tmp_path / "nested" / "dir" / "morning.lock"
        with _run_lock(lock_path):
            assert lock_path.exists()
