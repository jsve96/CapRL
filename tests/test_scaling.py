import pandas as pd
import pandas.testing as pdt

from caprl.data.scaling import RobustFeatureScaler


def test_scaler_parameters_depend_only_on_training_data():
    train_dates = pd.date_range("2020-01-01", periods=5)
    future_dates = pd.date_range("2020-01-06", periods=2)
    training = pd.DataFrame({"momentum": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=train_dates)
    future = pd.DataFrame({"momentum": [10.0, 20.0]}, index=future_dates)
    changed_future = pd.DataFrame({"momentum": [10_000.0, 20_000.0]}, index=future_dates)

    scaler = RobustFeatureScaler.fit(training, lower_quantile=0.0, upper_quantile=1.0)
    pdt.assert_frame_equal(scaler.transform(training), scaler.transform(training.copy()))
    assert scaler.fitted_end == train_dates[-1]
    assert scaler.transform(future).iloc[-1, 0] == scaler.transform(changed_future).iloc[-1, 0]


def test_constant_feature_has_safe_unit_scale():
    features = pd.DataFrame({"constant": [2.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2))
    scaler = RobustFeatureScaler.fit(features)
    assert scaler.scale["constant"] == 1.0
    assert (scaler.transform(features)["constant"] == 0.0).all()

