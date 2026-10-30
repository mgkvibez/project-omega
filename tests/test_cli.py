import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from omega.cli import main
from omega.vision.model import torch_available


def test_info_reports_the_stub_honestly(capsys):
    assert main(["info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["predictor"]["source"] == "stub"
    assert payload["predictor"]["reason"]
    assert isinstance(payload["torch_available"], bool)


def test_bench_json_works_without_torch(capsys):
    assert main(["bench", "--runs", "3", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs"] == 3 and payload["source"] == "stub" and payload["p50_ms"] >= 0


def test_no_subcommand_is_a_usage_error():
    assert main([]) == 2


@pytest.mark.skipif(torch_available(), reason="torch is installed here, so the refusal path cannot be hit")
def test_train_refuses_without_torch_instead_of_crashing(capsys):
    assert main(["train", "--out", "/tmp/never.pt"]) == 2
    assert "needs torch" in capsys.readouterr().err
