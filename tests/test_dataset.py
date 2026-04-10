import json
import pytest
from pathlib import Path
from llm_stress_test.dataset import load_dataset, DatasetError

class TestLoadDataset:
    def test_load_custom_jsonl(self, tmp_path):
        dataset_file = tmp_path / "custom.jsonl"
        lines = [json.dumps({"messages": [{"role": "user", "content": f"prompt {i}"}]}) for i in range(5)]
        dataset_file.write_text("\n".join(lines))
        prompts = load_dataset(str(dataset_file))
        assert len(prompts) == 5
        assert prompts[0] == {"role": "user", "content": "prompt 0"}

    def test_load_builtin_openqa(self):
        # Tests fallback behavior when dataset file doesn't exist yet
        prompts = load_dataset("openqa")
        assert len(prompts) > 0
        assert "role" in prompts[0]

    def test_nonexistent_file_raises(self):
        with pytest.raises(DatasetError, match="数据集不存在"):
            load_dataset("/nonexistent/dataset.jsonl")

    def test_invalid_format_raises(self, tmp_path):
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text("not json at all\n")
        with pytest.raises(DatasetError, match="格式错误"):
            load_dataset(str(bad_file))
