from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from src.preprocess import build_preprocessor


class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        preprocessor = build_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_has_two_named_transformers(self):
        preprocessor = build_preprocessor()
        names, transformers, columns = zip(*preprocessor.transformers)
        assert "target_enc" in names
        assert "ohe" in names
        assert len(names) == 2

    def test_target_encoder_columns(self):
        preprocessor = build_preprocessor()
        for name, transformer, cols in preprocessor.transformers:
            if name == "target_enc":
                assert isinstance(transformer, TargetEncoder)
                assert transformer.smooth == 5
                assert list(cols) == ["route", "booking_origin"]

    def test_onehot_encoder_columns(self):
        preprocessor = build_preprocessor()
        for name, transformer, cols in preprocessor.transformers:
            if name == "ohe":
                assert isinstance(transformer, OneHotEncoder)
                assert transformer.handle_unknown == "ignore"
                assert list(cols) == ["sales_channel", "trip_type", "flight_day"]

    def test_remainder_passthrough(self):
        preprocessor = build_preprocessor()
        remainder = preprocessor.remainder
        assert remainder == "passthrough"
