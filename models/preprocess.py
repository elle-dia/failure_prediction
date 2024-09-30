import pandas as pd

# CSVファイルの読み取り
# データはChatGPTを用いて作成
sensor_data = pd.read_csv("./data/sensor_data.csv")

# 移動平均を算出
sensor_data["Sensor1_Temperature_MA"] = sensor_data["Sensor1_Temperature_C"].rolling(window=3).mean()
sensor_data["Sensor2_Vibration_MA"] = sensor_data["Sensor2_Vibration_Hz"].rolling(window=3).mean()
sensor_data["Sensor3_Pressure_MA"] = sensor_data["Sensor3_Pressure_Pa"].rolling(window=3).mean()

# NaN値を削除
processed_sensor_data = sensor_data.dropna()

processed_sensor_data.to_csv("./data/processed_sensor_data.csv", index=False)
