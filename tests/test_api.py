"""
API Integration Tests
=====================
Tests all FastAPI endpoints:
  - GET /api/health
  - GET /api/model-info
  - GET /api/classes
  - GET /api/metrics
  - GET /api/inference-stats
  - POST /api/predict/video
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import sys
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "model_state" in data
    assert "device" in data


def test_model_info_endpoint(client):
    res = client.get("/api/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "backbone" in data
    assert "num_classes" in data
    assert "class_names" in data


def test_classes_endpoint(client):
    res = client.get("/api/classes")
    assert res.status_code == 200
    data = res.json()
    assert "class_names" in data
    assert len(data["class_names"]) > 0


def test_metrics_endpoint(client):
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "available" in data
    if data["available"]:
        assert "top_1_accuracy" in data
        assert "fps_estimate" in data


def test_inference_stats_endpoint(client):
    res = client.get("/api/inference-stats")
    assert res.status_code == 200
    data = res.json()
    assert "average_inference_ms" in data
    assert "fps" in data


def test_predict_video_endpoint(client):
    test_video_path = Path(__file__).resolve().parent.parent / "data/ucf101/JumpingJack/v_JumpingJack_g01_c01.mp4"
    if not test_video_path.exists():
        pytest.skip("Test video file does not exist")

    with open(test_video_path, "rb") as f:
        files = {"file": ("video.mp4", f, "video/mp4")}
        res = client.post("/api/predict/video?use_pose=true", files=files)

    assert res.status_code == 200
    data = res.json()
    assert "activity" in data
    assert "confidence" in data
    assert "top_predictions" in data
    assert "fps" in data
    assert len(data["top_predictions"]) > 0
