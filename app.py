import streamlit as st
import google.generativeai as genai
import requests
from datetime import datetime, timedelta
from PIL import Image
import io
import os
from dotenv import load_dotenv
import json
import pandas as pd
import plotly.graph_objects as go
from streamlit_geolocation import streamlit_geolocation

# Load environment variables
load_dotenv()

# Configure APIs
# Try to get from Streamlit secrets first, then fall back to environment variable
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
except:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Page configuration
st.set_page_config(
    page_title="作物生育予測アプリ",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 作物生育予測アプリ")
st.markdown("気温データとAIを使って、作物の収穫時期を予測します")

# Sidebar for API key configuration
with st.sidebar:
    st.header("⚙️ 設定")

    # API Key inputs (if not in .env)
    if not GEMINI_API_KEY:
        gemini_key_input = st.text_input("Gemini API Key", type="password")
        if gemini_key_input:
            GEMINI_API_KEY = gemini_key_input
            genai.configure(api_key=GEMINI_API_KEY)
    else:
        st.success("✓ Gemini API設定済み")

    st.info("🌤️ Open-Meteo API使用中（無料・APIキー不要）")

    st.markdown("---")
    st.markdown("### 使い方")
    st.markdown("""
    1. 位置情報を設定
    2. 作物の画像をアップロード
    3. 追加情報を入力（任意）
    4. 予測を実行
    """)


def get_coordinates_from_location(location_str):
    """
    位置情報文字列から緯度経度を取得
    """
    # すでに緯度経度の形式（例: "35.6762,139.6503"）の場合
    if ',' in location_str and not any(c.isalpha() for c in location_str):
        try:
            lat, lon = map(float, location_str.split(','))
            return lat, lon
        except:
            pass

    # 地名の場合はOpen-MeteoのGeocoding APIを使用
    try:
        geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            'name': location_str,
            'count': 1,
            'language': 'ja',
            'format': 'json'
        }
        response = requests.get(geocoding_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                return data['results'][0]['latitude'], data['results'][0]['longitude']
    except Exception as e:
        st.error(f"位置情報の取得に失敗しました: {str(e)}")

    return None, None


def get_weather_data(location_str):
    """
    Open-Meteo APIから過去と未来の気象データを取得
    """
    try:
        # 緯度経度を取得
        latitude, longitude = get_coordinates_from_location(location_str)

        if latitude is None or longitude is None:
            st.error("位置情報を取得できませんでした。")
            return None

        # 過去7日間のデータ
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        history_url = "https://archive-api.open-meteo.com/v1/archive"
        history_params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date,
            'end_date': end_date,
            'daily': 'temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,precipitation_sum',
            'timezone': 'auto'
        }

        history_response = requests.get(history_url, params=history_params)
        history_data = history_response.json() if history_response.status_code == 200 else None

        # 未来7日間の予報
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        forecast_params = {
            'latitude': latitude,
            'longitude': longitude,
            'daily': 'temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,precipitation_sum',
            'timezone': 'auto',
            'forecast_days': 7
        }

        forecast_response = requests.get(forecast_url, params=forecast_params)
        forecast_data = forecast_response.json() if forecast_response.status_code == 200 else None

        return {
            'history': history_data,
            'forecast': forecast_data
        }
    except Exception as e:
        st.error(f"気象データの取得に失敗しました: {str(e)}")
        return None


def analyze_crop_image(image, additional_info=""):
    """
    Gemini APIを使って作物の画像を分析
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""
この画像を分析して、以下の情報を日本語のJSON形式で返してください：

1. crop_type: 作物の種類（例: トマト、イチゴ、レタスなど）
2. growth_stage: 現在の成長段階（例: 発芽期、生育期、開花期、結実期など）
3. health_status: 健康状態（良好、注意が必要、問題ありなど）
4. estimated_progress: 成長の進捗度合い（0-100%）
5. observations: その他の観察事項

追加情報: {additional_info if additional_info else "なし"}

必ずJSON形式で返してください。
"""

        response = model.generate_content([prompt, image])

        # JSONを抽出
        response_text = response.text
        # マークダウンのコードブロックを削除
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        return json.loads(response_text)
    except Exception as e:
        st.error(f"画像分析に失敗しました: {str(e)}")
        return None


def predict_harvest_date(crop_analysis, weather_data, additional_info=""):
    """
    Gemini APIを使って収穫時期を予測
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        # 気象データを要約
        weather_summary = create_weather_summary(weather_data)

        prompt = f"""
以下の情報をもとに、作物の収穫時期を予測してください：

【作物分析結果】
{json.dumps(crop_analysis, ensure_ascii=False, indent=2)}

【気象データ】
{weather_summary}

【追加情報】
{additional_info if additional_info else "なし"}

以下の情報を日本語のJSON形式で返してください：
1. estimated_harvest_date: 予測収穫日（YYYY-MM-DD形式）
2. confidence_level: 予測の信頼度（高い、中程度、低いのいずれか）
3. days_until_harvest: 収穫までの予測日数
4. reasoning: 予測の根拠（簡潔に）
5. recommendations: 栽培上の推奨事項（リスト形式）
6. risk_factors: リスク要因（リスト形式）

必ずJSON形式で返してください。
"""

        response = model.generate_content(prompt)

        # JSONを抽出
        response_text = response.text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        return json.loads(response_text)
    except Exception as e:
        st.error(f"収穫時期の予測に失敗しました: {str(e)}")
        return None


def create_weather_summary(weather_data):
    """
    気象データを要約してテキスト形式で返す
    """
    summary = "【過去7日間の気象】\n"

    if weather_data and 'history' in weather_data and weather_data['history']:
        history = weather_data['history']
        if 'daily' in history:
            daily = history['daily']
            for i in range(len(daily['time'])):
                date = daily['time'][i]
                max_temp = daily['temperature_2m_max'][i]
                min_temp = daily['temperature_2m_min'][i]
                humidity = daily['relative_humidity_2m_mean'][i]
                summary += f"{date}: 最高気温{max_temp:.1f}°C, 最低気温{min_temp:.1f}°C, 平均湿度{humidity:.0f}%\n"

    summary += "\n【今後7日間の予報】\n"

    if weather_data and 'forecast' in weather_data and weather_data['forecast']:
        forecast = weather_data['forecast']
        if 'daily' in forecast:
            daily = forecast['daily']
            for i in range(len(daily['time'])):
                date = daily['time'][i]
                max_temp = daily['temperature_2m_max'][i]
                min_temp = daily['temperature_2m_min'][i]
                humidity = daily['relative_humidity_2m_mean'][i]
                summary += f"{date}: 最高気温{max_temp:.1f}°C, 最低気温{min_temp:.1f}°C, 平均湿度{humidity:.0f}%\n"

    return summary


def plot_temperature_chart(weather_data):
    """
    気温の推移をグラフ化
    """
    dates = []
    max_temps = []
    min_temps = []
    avg_humidity = []

    # 過去のデータ
    if weather_data and 'history' in weather_data and weather_data['history']:
        history = weather_data['history']
        if 'daily' in history:
            daily = history['daily']
            for i in range(len(daily['time'])):
                dates.append(daily['time'][i])
                max_temps.append(daily['temperature_2m_max'][i])
                min_temps.append(daily['temperature_2m_min'][i])
                avg_humidity.append(daily['relative_humidity_2m_mean'][i])

    # 未来のデータ
    if weather_data and 'forecast' in weather_data and weather_data['forecast']:
        forecast = weather_data['forecast']
        if 'daily' in forecast:
            daily = forecast['daily']
            for i in range(len(daily['time'])):
                dates.append(daily['time'][i])
                max_temps.append(daily['temperature_2m_max'][i])
                min_temps.append(daily['temperature_2m_min'][i])
                avg_humidity.append(daily['relative_humidity_2m_mean'][i])

    # グラフ作成
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=max_temps,
        mode='lines+markers',
        name='最高気温',
        line=dict(color='red', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=min_temps,
        mode='lines+markers',
        name='最低気温',
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title='気温の推移',
        xaxis_title='日付',
        yaxis_title='気温 (°C)',
        hovermode='x unified'
    )

    return fig


# Main UI
st.header("📍 ステップ1: 位置情報の設定")

col1, col2 = st.columns(2)

with col1:
    st.subheader("GPS位置情報を取得")
    location_data = streamlit_geolocation()

    if location_data and location_data.get("latitude"):
        st.success(f"✓ 位置情報取得済み")
        st.write(f"緯度: {location_data['latitude']:.4f}, 経度: {location_data['longitude']:.4f}")
        location_str = f"{location_data['latitude']},{location_data['longitude']}"
    else:
        location_str = None
        st.info("「位置情報を取得」ボタンをクリックしてください")

with col2:
    st.subheader("または地名を入力")
    manual_location = st.text_input("地名または住所", placeholder="例: Tokyo, 東京都渋谷区")
    if manual_location:
        location_str = manual_location
        st.success(f"✓ 地名設定: {manual_location}")

st.markdown("---")

# Image upload and analysis
st.header("🌿 ステップ2: 作物の情報入力")

uploaded_file = st.file_uploader("作物の画像をアップロード", type=['jpg', 'jpeg', 'png'])

additional_text = st.text_area(
    "追加情報（任意）",
    placeholder="例: 種まきから2週間経過、葉が少し黄色くなっている、など",
    height=100
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="アップロードされた画像", use_column_width=True)

    with col2:
        st.info("画像がアップロードされました。「予測を実行」ボタンをクリックしてください。")

st.markdown("---")

# Prediction button
st.header("🔮 ステップ3: 予測の実行")

if st.button("🚀 収穫時期を予測", type="primary", use_container_width=True):
    if not GEMINI_API_KEY:
        st.error("❌ Gemini APIキーが設定されていません。サイドバーで設定してください。")
    elif not location_str:
        st.error("❌ 位置情報が設定されていません。")
    elif not uploaded_file:
        st.error("❌ 作物の画像をアップロードしてください。")
    else:
        with st.spinner("分析中..."):
            # 1. 画像分析
            st.info("🔍 画像を分析しています...")
            image = Image.open(uploaded_file)
            crop_analysis = analyze_crop_image(image, additional_text)

            if crop_analysis:
                st.success("✓ 画像分析完了")

                # 2. 気象データ取得
                st.info("🌤️ 気象データを取得しています...")
                weather_data = get_weather_data(location_str)

                if weather_data:
                    st.success("✓ 気象データ取得完了")

                    # 3. 収穫予測
                    st.info("🤖 収穫時期を予測しています...")
                    harvest_prediction = predict_harvest_date(crop_analysis, weather_data, additional_text)

                    if harvest_prediction:
                        st.success("✓ 予測完了！")

                        st.markdown("---")
                        st.header("📊 分析結果")

                        # 作物分析結果
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("🌱 作物分析")
                            st.metric("作物の種類", crop_analysis.get('crop_type', '不明'))
                            st.metric("成長段階", crop_analysis.get('growth_stage', '不明'))
                            st.metric("健康状態", crop_analysis.get('health_status', '不明'))
                            st.metric("成長進捗", f"{crop_analysis.get('estimated_progress', 0)}%")

                            if crop_analysis.get('observations'):
                                st.write("**観察事項:**")
                                st.write(crop_analysis['observations'])

                        with col2:
                            st.subheader("📅 収穫予測")
                            st.metric(
                                "予測収穫日",
                                harvest_prediction.get('estimated_harvest_date', '不明'),
                                delta=f"{harvest_prediction.get('days_until_harvest', '?')}日後"
                            )
                            st.metric("信頼度", harvest_prediction.get('confidence_level', '不明'))

                            st.write("**予測の根拠:**")
                            st.write(harvest_prediction.get('reasoning', ''))

                        # 推奨事項とリスク要因
                        st.markdown("---")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("💡 推奨事項")
                            recommendations = harvest_prediction.get('recommendations', [])
                            if recommendations:
                                for rec in recommendations:
                                    st.write(f"• {rec}")
                            else:
                                st.write("特になし")

                        with col2:
                            st.subheader("⚠️ リスク要因")
                            risk_factors = harvest_prediction.get('risk_factors', [])
                            if risk_factors:
                                for risk in risk_factors:
                                    st.write(f"• {risk}")
                            else:
                                st.write("特になし")

                        # 気温グラフ
                        st.markdown("---")
                        st.subheader("🌡️ 気温の推移")
                        temp_chart = plot_temperature_chart(weather_data)
                        st.plotly_chart(temp_chart, use_container_width=True)

st.markdown("---")
st.caption("© 2024 作物生育予測アプリ | Powered by Gemini AI & Open-Meteo")
