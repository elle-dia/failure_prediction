import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

sensor_data = pd.read_csv("./data/processed_sensor_data.csv")

# 必要な特徴量の選択（移動平均のカラムを使用）
features = ["Sensor1_Temperature_MA", "Sensor2_Vibration_MA", "Sensor3_Pressure_MA"]
X = sensor_data[features].dropna()  # 欠損値を削除

# モデル定義
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

# モデルの学習
iso_forest.fit(X)

# 異常検知の予測 (1: 正常, -1: 異常)
sensor_data["anomaly"] = iso_forest.predict(X)

# モデルの保存
joblib.dump(iso_forest, "./models/isolation_forest_model.pkl")
