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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

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

    if not WEATHER_API_KEY:
        weather_key_input = st.text_input("WeatherAPI Key", type="password")
        if weather_key_input:
            WEATHER_API_KEY = weather_key_input
    else:
        st.success("✓ WeatherAPI設定済み")

    st.markdown("---")
    st.markdown("### 使い方")
    st.markdown("""
    1. 位置情報を設定
    2. 作物の画像をアップロード
    3. 追加情報を入力（任意）
    4. 予測を実行
    """)


def get_weather_data(location, api_key):
    """
    WeatherAPIから過去と未来の気象データを取得
    """
    try:
        # 過去7日間のデータ
        history_data = []
        for i in range(7, 0, -1):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            url = f"http://api.weatherapi.com/v1/history.json"
            params = {
                'key': api_key,
                'q': location,
                'dt': date
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                history_data.append(response.json())

        # 未来7日間の予報
        forecast_url = f"http://api.weatherapi.com/v1/forecast.json"
        forecast_params = {
            'key': api_key,
            'q': location,
            'days': 7
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
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

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
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

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

    if weather_data and 'history' in weather_data:
        for day_data in weather_data['history']:
            date = day_data['forecast']['forecastday'][0]['date']
            day = day_data['forecast']['forecastday'][0]['day']
            summary += f"{date}: 最高気温{day['maxtemp_c']}°C, 最低気温{day['mintemp_c']}°C, 平均湿度{day['avghumidity']}%\n"

    summary += "\n【今後7日間の予報】\n"

    if weather_data and 'forecast' in weather_data and weather_data['forecast']:
        for day in weather_data['forecast']['forecast']['forecastday']:
            date = day['date']
            day_data = day['day']
            summary += f"{date}: 最高気温{day_data['maxtemp_c']}°C, 最低気温{day_data['mintemp_c']}°C, 平均湿度{day_data['avghumidity']}%\n"

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
    if weather_data and 'history' in weather_data:
        for day_data in weather_data['history']:
            date = day_data['forecast']['forecastday'][0]['date']
            day = day_data['forecast']['forecastday'][0]['day']
            dates.append(date)
            max_temps.append(day['maxtemp_c'])
            min_temps.append(day['mintemp_c'])
            avg_humidity.append(day['avghumidity'])

    # 未来のデータ
    if weather_data and 'forecast' in weather_data and weather_data['forecast']:
        for day in weather_data['forecast']['forecast']['forecastday']:
            dates.append(day['date'])
            max_temps.append(day['day']['maxtemp_c'])
            min_temps.append(day['day']['mintemp_c'])
            avg_humidity.append(day['day']['avghumidity'])

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
    if not GEMINI_API_KEY or not WEATHER_API_KEY:
        st.error("❌ APIキーが設定されていません。サイドバーで設定してください。")
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
                weather_data = get_weather_data(location_str, WEATHER_API_KEY)

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
st.caption("© 2024 作物生育予測アプリ | Powered by Gemini AI & WeatherAPI")
