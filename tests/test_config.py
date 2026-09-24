"""Muat dan validasi configs/experiment.yaml."""

import copy

import pytest
import yaml

from pcdface.config import ConfigError, HaarConfig, MediaPipeConfig, YCbCrConfig, parse_config
from pcdface.paths import CONFIG_PATH, PROJECT_ROOT


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_project_config_loads(cfg):
    assert cfg.seed == 42
    assert isinstance(cfg.detector("haar"), HaarConfig)
    assert isinstance(cfg.detector("mp_short"), MediaPipeConfig)
    assert isinstance(cfg.detector("ycbcr"), YCbCrConfig)
    assert cfg.paths.raw == PROJECT_ROOT / "data" / "raw"
    assert cfg.evaluation.iou_thresholds == (0.3, 0.4, 0.5)
    assert cfg.scale_for((640, 360)) == 0.5
    assert cfg.dataset.formations["F5"] == (80, 150, 250)


def test_ap_run_is_not_a_detector(cfg):
    # PRD §12.2: ap_score_floor dipindah dari detectors ke evaluation.ap_run
    assert "ap_score_floor" not in cfg.detectors
    assert cfg.evaluation.ap_run.haar_min_neighbors == 0


def test_unknown_detector_name_raises(cfg):
    with pytest.raises(ConfigError, match="tidak ada di config"):
        cfg.detector("yunet")


def test_all_errors_reported_together(raw):
    broken = copy.deepcopy(raw)
    broken["detectors"]["haar"]["scale_factor"] = 0.9
    broken["evaluation"]["iou_primary"] = 1.5
    del broken["stats"]["ci_level"]
    with pytest.raises(ConfigError) as info:
        parse_config(broken)
    message = str(info.value)
    assert "detectors.haar.scale_factor" in message
    assert "evaluation.iou_primary" in message
    assert "stats.ci_level: wajib ada" in message


def test_experiment_must_reference_known_detector(raw):
    broken = copy.deepcopy(raw)
    broken["experiments"]["e2"]["detectors"] = ["haar", "tidak_ada"]
    with pytest.raises(ConfigError, match="'tidak_ada' tidak ada di bagian detectors"):
        parse_config(broken)


def test_resolution_must_keep_aspect_ratio(raw):
    broken = copy.deepcopy(raw)
    broken["experiments"]["e1"]["resolutions"] = [[1280, 720], [640, 480]]
    with pytest.raises(ConfigError, match="rasio aspek"):
        parse_config(broken)


def test_reference_distance_must_be_a_captured_distance(raw):
    broken = copy.deepcopy(raw)
    broken["dataset"]["reference_distance_cm"] = 120
    with pytest.raises(ConfigError, match="reference_distance_cm"):
        parse_config(broken)


def test_bool_is_not_accepted_as_number(raw):
    broken = copy.deepcopy(raw)
    broken["detectors"]["haar"]["min_neighbors"] = True
    with pytest.raises(ConfigError, match="min_neighbors"):
        parse_config(broken)
