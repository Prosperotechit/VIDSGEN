import asyncio

import pytest

from app.core.config import APP_DIR, Settings
from app.storage import InvalidStorageKey, LocalStorage, ObjectNotFound, output_key, upload_key


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(tmp_path / "store")


def test_put_get_exists_size_delete(storage):
    key = "outputs/job_1/video.mp4"
    assert not run(storage.exists(key))
    run(storage.put_bytes(key, b"hello world", "video/mp4"))
    assert run(storage.exists(key))
    assert run(storage.get_bytes(key)) == b"hello world"
    assert run(storage.size(key)) == 11
    run(storage.delete(key))
    assert not run(storage.exists(key))
    assert not (storage.root / "outputs").exists()   # empty parent dirs tidied up
    run(storage.delete(key))                          # deleting twice is fine


def test_overwrite_is_atomic_and_leaves_no_temp_files(storage):
    key = "uploads/job_2/input.png"
    run(storage.put_bytes(key, b"v1", "image/png"))
    run(storage.put_bytes(key, b"v2", "image/png"))
    assert run(storage.get_bytes(key)) == b"v2"
    assert [p.name for p in (storage.root / "uploads" / "job_2").iterdir()] == ["input.png"]


def test_put_file_copies_source(storage, tmp_path):
    src = tmp_path / "source.mp4"
    src.write_bytes(b"x" * 3_000_000)
    run(storage.put_file("outputs/j/video.mp4", src, "video/mp4"))
    assert src.exists()
    assert run(storage.size("outputs/j/video.mp4")) == 3_000_000


def test_iter_chunks(storage):
    run(storage.put_bytes("a/b.bin", bytes(range(256)) * 10, "application/octet-stream"))

    async def collect():
        return [c async for c in storage.iter_chunks("a/b.bin", chunk_size=1000)]

    chunks = run(collect())
    assert [len(c) for c in chunks] == [1000, 1000, 560]
    assert b"".join(chunks) == bytes(range(256)) * 10


def test_missing_objects_raise(storage):
    with pytest.raises(ObjectNotFound):
        run(storage.get_bytes("nope/x"))
    with pytest.raises(ObjectNotFound):
        run(storage.size("nope/x"))
    assert storage.local_path("nope/x") is None


def test_local_path_inside_root(storage):
    run(storage.put_bytes("outputs/j/video.mp4", b"1", "video/mp4"))
    p = storage.local_path("outputs/j/video.mp4")
    assert p.is_file() and storage.root in p.parents


@pytest.mark.parametrize("key", [
    "", "/etc/passwd", "../escape", "a/../../b", "a/./b", "a//b", ".hidden", "a/.hidden",
    "a\\b", "with space", "trailing/", "x" * 600,
])
def test_invalid_keys_rejected(storage, key):
    with pytest.raises(InvalidStorageKey):
        run(storage.put_bytes(key, b"x", "text/plain"))
    assert not any(storage.root.parent.glob("escape*"))


def test_key_helpers():
    assert upload_key("job_abc", "png") == "uploads/job_abc/input.png"
    assert output_key("job_abc") == "outputs/job_abc/video.mp4"
    with pytest.raises(InvalidStorageKey):
        output_key("../job")


def test_health(storage):
    assert run(storage.healthy()) is True


def test_storage_root_inside_source_tree_is_refused():
    with pytest.raises(ValueError, match="must not be inside the application source"):
        Settings(_env_file=None, storage_local_root=APP_DIR / "videos")


def test_env_example_yields_safe_defaults(monkeypatch, tmp_path):
    """`cp .env.example .env` must work as-is: empty values fall back to safe defaults
    (not the current working directory)."""
    from app.core.config import REPO_ROOT

    monkeypatch.chdir(tmp_path)
    s = Settings(_env_file=REPO_ROOT / ".env.example")
    assert s.storage_local_root == REPO_ROOT / "storage"
    assert s.database_path == REPO_ROOT / "storage" / "vidsgen.db"
    assert s.inference_provider == "mock" and s.storage_backend == "local"
    assert s.cors_origin_list == ["http://localhost:5173"]
    assert s.comfyui_api_token.get_secret_value() == ""


def test_default_storage_root_is_outside_source_and_git_ignored():
    import subprocess

    s = Settings(_env_file=None)
    assert APP_DIR not in s.storage_local_root.parents
    probe = s.storage_local_root / "outputs" / "job" / "video.mp4"
    r = subprocess.run(["git", "check-ignore", "-q", str(probe)], cwd=APP_DIR)
    assert r.returncode == 0, "default storage location must be git-ignored"
