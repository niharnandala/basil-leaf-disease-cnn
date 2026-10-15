"""Config and class-order tests. These run without TensorFlow installed."""

from src import config as config_module


def test_config_loads():
    cfg = config_module.load()
    assert cfg.image_size == 224
    assert cfg.batch_size == 32
    assert 0 < cfg.validation_split < 1


def test_classes_are_alphabetical():
    """flow_from_directory sorts class folders, so the order here must match
    or every prediction gets the wrong label."""
    cfg = config_module.load()
    assert cfg.classes == sorted(cfg.classes)
    assert len(cfg.classes) == 4


def test_model_params_are_consistent():
    cfg = config_module.load()
    params = cfg.model_params
    # one dropout rate per dense layer, or the model builder silently drops a layer
    assert len(params["dense_units"]) == len(params["dropout"])
    assert all(0 <= rate < 1 for rate in params["dropout"])
