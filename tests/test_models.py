import pytest
from src.models import ModelConfig, validate_features, get_rf_model, get_xgb_model


class TestModelConfig:
    def test_default_values(self):
        cfg = ModelConfig()
        assert cfg.n_estimators == 400
        assert cfg.learning_rate == 0.01
        assert cfg.max_depth == 6
        assert cfg.random_state == 42
        assert cfg.n_jobs >= 1

    def test_all_features_property(self):
        cfg = ModelConfig()
        expected = cfg.categorical_features + cfg.numeric_features
        assert cfg.all_features == expected

    def test_frozen_config_cannot_be_modified(self):
        cfg = ModelConfig()
        with pytest.raises(Exception):
            cfg.n_estimators = 500

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            ModelConfig(invalid_field=True)

    def test_n_estimators_validation(self):
        with pytest.raises(Exception):
            ModelConfig(n_estimators=0)
        with pytest.raises(Exception):
            ModelConfig(n_estimators=2001)

    def test_learning_rate_validation(self):
        with pytest.raises(Exception):
            ModelConfig(learning_rate=0)
        with pytest.raises(Exception):
            ModelConfig(learning_rate=1)

    def test_max_depth_validation(self):
        with pytest.raises(Exception):
            ModelConfig(max_depth=0)
        with pytest.raises(Exception):
            ModelConfig(max_depth=21)


class TestValidateFeatures:
    def test_all_features_present(self):
        cfg = ModelConfig()
        cols = cfg.all_features
        validate_features(cols, cfg)

    def test_missing_features_raises(self):
        cfg = ModelConfig()
        with pytest.raises(ValueError, match="missing mandatory columns"):
            validate_features(["route"], cfg)


class TestModelFactories:
    def test_get_rf_model_returns_random_forest(self):
        cfg = ModelConfig()
        model = get_rf_model(cfg)
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 400
        assert model.class_weight == "balanced"

    def test_get_xgb_model_returns_xgb_classifier(self):
        cfg = ModelConfig()
        model = get_xgb_model(cfg)
        from xgboost import XGBClassifier
        assert isinstance(model, XGBClassifier)
        assert model.n_estimators == 400
        assert model.get_params()["eval_metric"] == "auc"
