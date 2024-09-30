import os

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.ensemble import IsolationForest

load_dotenv()

MODEL_PATH = "./models/isolation_forest_model.pkl"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
REQUIRED_COLUMNS = ["Sensor1_Temperature_C", "Sensor2_Vibration_Hz", "Sensor3_Pressure_Pa"]
ROLLING_WINDOW = 3
SYSTEM_PROMPT = """あなたは機器の故障予測についてレポートを作成するAIアシスタントです。
あなたがレポートを作成するために以下のような異常の検知データがCSVフォーマットで提供されます。

```
Timestamp,Sensor1_Temperature_C,Sensor2_Vibration_Hz,Sensor3_Pressure_Pa,Sensor1_Temperature_MA,Sensor2_Vibration_MA,Sensor3_Pressure_MA,Prediction,Result
2024-09-02 01:00:00,76.11,22.18,1009.19,68.80666666666667,27.176666666666666,930.9233333333333,-1,異常
2024-09-02 16:00:00,99.46,38.54,1218.07,79.38333333333333,28.3,1042.78,-1,異常
2024-09-02 17:00:00,76.71,17.71,979.54,84.38,27.16333333333333,1036.3866666666665,-1,異常
2024-09-02 18:00:00,73.84,25.39,973.8,83.33666666666666,27.213333333333335,1057.1366666666665,-1,異常
2024-09-03 08:00:00,99.81,43.8,1220.1,89.81,25.896666666666665,1037.2833333333335,-1,異常
2024-09-03 09:00:00,71.91,25.86,1043.72,85.34333333333332,28.766666666666666,1074.2866666666666,-1,異常
2024-09-03 10:00:00,78.31,22.72,1013.13,83.34333333333332,30.793333333333333,1092.3166666666668,-1,異常
2024-09-03 13:00:00,96.37,40.55,1218.94,91.10333333333334,30.97333333333333,1158.4966666666667,-1,異常
2024-09-03 14:00:00,63.94,25.74,1019.52,84.16333333333334,33.50333333333333,1161.7766666666666,-1,異常
```

機器の異常の原因としては以下が挙げられます。
- 温度異常: 温度が90°Cを超える場合、冷却装置の異常や過熱のリスクがある。
- 振動異常: 振動が30Hzを超える場合、部品の摩耗や不安定な動作が考えられる。
- 圧力異常: 圧力が1200Paを超える場合、システムの過負荷や圧力調整機構の故障が考えられる。

機器は以下の条件だと、故障する可能性が高いです。
- 24時間以内に5回以上異常が起きている
    - 24時間以内というのは例えば、2024-09-03 00:00:00から2024-09-04 00:00:00までです。

以上の情報を用いて、提供されたデータを記録した機器の故障の可能性とその理由についてのレポートを作成してください。
"""
INPUT_PROMPT = "以下の異常データに基づいて、機器の故障に関する詳細なレポートを生成してください。\n\n異常データ:\n{}"


@st.cache_resource
def load_model(path: str) -> IsolationForest:
    """異常検知モデルをロードする"""
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"モデルのロードに失敗しました: {e}")
        st.stop()


@st.cache_resource
def get_azure_openai_client(api_key: str, endpoint: str) -> AzureOpenAI:
    """Azure OpenAIクライアントを取得する"""
    try:
        return AzureOpenAI(api_key=api_key, api_version="2024-07-01-preview", azure_endpoint=endpoint)
    except Exception as e:
        st.error(f"OpenAIクライアントの初期化に失敗しました: {e}")
        st.stop()


def compute_moving_average(data: pd.DataFrame, columns: list, window: int = 3) -> pd.DataFrame:
    """指定された列の移動平均を計算して欠損値を削除する"""
    for col in columns:
        # "_C", "_Hz", "_Pa" の部分を削除して "_MA" を追加
        base_name = "_".join(col.split("_")[:-1])
        ma_col = f"{base_name}_MA"
        data[ma_col] = data[col].rolling(window=window).mean()
    return data.dropna()


def predict_anomalies(model: IsolationForest, data: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """異常を予測し、結果をデータフレームに追加する"""
    X = data[feature_columns]
    predictions = model.predict(X)
    data["Prediction"] = predictions
    data["Result"] = data["Prediction"].apply(lambda x: "正常" if x == 1 else "異常")
    return data


def plot_sensor_data(
    data: pd.DataFrame,
    anomalies: pd.DataFrame,
    timestamp_col: str,
    sensor_col: str,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    """センサーデータをプロットする"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data[timestamp_col], data[sensor_col], label=ylabel, color=color)
    ax.scatter(
        anomalies[timestamp_col], anomalies[sensor_col], color="red", label="Anomaly", zorder=5
    )  # Anomaly を赤点で表示
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    fig.autofmt_xdate()
    ax.legend()
    st.pyplot(fig)


def generate_report(aoai_client: AzureOpenAI, anomalies: pd.DataFrame) -> str:
    """GPTを使用してレポートを生成する"""
    try:
        first_response = aoai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": INPUT_PROMPT.format(anomalies.to_string())},
            ],
            temperature=0.0,
        )
        if first_response.choices[0].finish_reason == "stop":
            return first_response.choices[0].message.content
        else:
            # 初期のmessagesに最初の応答を含むリストを設定
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": INPUT_PROMPT.format(anomalies.to_string())},
                {"role": "assistant", "content": first_response.choices[0].message.content},
            ]
            result = first_response.choices[0].message.content

            while True:
                response = aoai_client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=messages,  # ここでメッセージ履歴を含む
                    temperature=0.0,
                )
                # 新しいレスポンスをmessagesに追加
                messages.append({"role": "assistant", "content": response.choices[0].message.content})
                result += response.choices[0].message.content

                # finish_reasonが "stop" になったらループを終了
                if response.choices[0].finish_reason == "stop":
                    return result

                # ユーザーからの応答として、生成されたレスポンスをmessagesに追加
                messages.append({"role": "user", "content": response.choices[0].message.content})

    except Exception as e:
        st.error(f"レポートの生成に失敗しました: {e}")
        return "レポートの生成に失敗しました。"


def main():
    """メイン関数"""
    st.title("故障予測デモアプリ")
    st.write(
        "このリポジトリは、機械学習モデル（Isolation Forest）と LLM を組み合わせた故障予測デモアアプリケーションです。"
    )
    uploaded_file = st.file_uploader("対象のCSVファイルをアップロードしてください", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"CSVファイルの読み込みに失敗しました: {e}")
            st.stop()

        st.subheader("アップロードされたデータのプレビュー")
        st.write(data.head())

        missing_columns = [col for col in REQUIRED_COLUMNS if col not in data.columns]

        if not missing_columns:
            with st.spinner("AIが異常の検知とレポートの作成を行っています..."):
                data = compute_moving_average(data, REQUIRED_COLUMNS, window=ROLLING_WINDOW)
                model = load_model(MODEL_PATH)
                # 修正: feature_columns を正しい名前に設定
                feature_columns = [f"{'_'.join(col.split('_')[:-1])}_MA" for col in REQUIRED_COLUMNS]
                data = predict_anomalies(model, data, feature_columns)

                st.subheader("AIによる機器の異常検知の結果")
                st.write(data)
                csv = data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="予測結果をダウンロード",
                    data=csv,
                    file_name="prediction_results.csv",
                    mime="text/csv",
                )

                anomalies = data[data["Result"] == "異常"]
                try:
                    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
                except Exception as e:
                    st.error(f"Timestamp列の変換に失敗しました: {e}")
                    st.stop()

                plot_sensor_data(
                    data,
                    anomalies,
                    "Timestamp",
                    "Sensor1_Temperature_C",
                    "Sensor 1 Temperature Data with Anomalies",
                    "Temperature (°C)",
                    "blue",
                )
                plot_sensor_data(
                    data,
                    anomalies,
                    "Timestamp",
                    "Sensor2_Vibration_Hz",
                    "Sensor 2 Vibration Data with Anomalies",
                    "Vibration (Hz)",
                    "green",
                )
                plot_sensor_data(
                    data,
                    anomalies,
                    "Timestamp",
                    "Sensor3_Pressure_Pa",
                    "Sensor 3 Pressure Data with Anomalies",
                    "Pressure (Pa)",
                    "purple",
                )

                aoai_client = get_azure_openai_client(AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)
                report = generate_report(aoai_client, anomalies)

                st.subheader("AIによる故障予測レポート")
                st.write(report)

                # テキストファイルとしてレポートをダウンロード
                report_file = report.encode("utf-8")  # テキストをUTF-8にエンコード
                st.download_button(
                    label="レポートをダウンロード",
                    data=report_file,
                    file_name="failure_report.txt",
                    mime="text/plain",
                )

        else:
            st.error(
                f"アップロードされたCSVに必要なセンサー列が含まれていません。\n不足している列: {', '.join(missing_columns)}"
            )


if __name__ == "__main__":
    main()
