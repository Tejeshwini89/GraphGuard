from graphguard.config import load_config


def test_config_has_required_sections():
    config = load_config()

    assert config["project"]["name"] == "GraphGuard"
    assert config["dataset"]["name"] == "elliptic-bitcoin"
    assert config["splits"]["strategy"] == "temporal"
    assert config["baseline"]["model"] == "xgboost"
    assert config["models"]["graphsage"]["layers"] == 2
    assert config["models"]["gat"]["heads"] == 4
    assert config["evaluation"]["primary_metric"] == "pr_auc"
