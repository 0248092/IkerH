from __future__ import annotations
import warnings
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import google.generativeai as genai
import os
import time

warnings.filterwarnings("ignore")

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(
    page_title="Análisis de Portafolio de Acciones",
    layout="wide",
    page_icon="📊"
)

# Configurar Gemini
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    st.error("❌ No se encontró GEMINI_API_KEY")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# =========================
# ESTILOS CSS
# =========================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stock-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 2px solid #e9ecef;
    }
    .ai-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# FUNCIONES CON RATE LIMITING
# =========================

@st.cache_data(ttl=3600)
def scrape_yfinance_data(ticker: str) -> Dict:
    """Obtiene datos de Yahoo Finance con manejo de rate limiting"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Delay para evitar rate limiting
            time.sleep(1)
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Verificar si obtuvimos datos válidos
            if not info or 'symbol' not in info:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return {}
            
            data = {
                'nombre': info.get('longName', info.get('shortName', ticker)),
                'sector': info.get('sector', 'N/D'),
                'industria': info.get('industry', 'N/D'),
                'precio_actual': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'eps': info.get('trailingEps', 0),
                'precio_objetivo': info.get('targetMeanPrice', 0)
            }
            
            return data
            
        except Exception as e:
            if "Too Many Requests" in str(e) or "429" in str(e):
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    st.warning(f"⏳ Rate limit alcanzado para {ticker}. Esperando {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            if attempt == max_retries - 1:
                st.warning(f"⚠️ No se pudieron obtener datos completos de {ticker}")
                return {}
    
    return {}

@st.cache_data(ttl=3600)
def scrape_finviz_data(ticker: str) -> Dict:
    """Obtiene datos de Finviz con rate limiting"""
    try:
        time.sleep(0.5)  # Delay para evitar bloqueos
        
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return {}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'snapshot-table2'})
        data = {}
        
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                for i in range(0, len(cols), 2):
                    if i+1 < len(cols):
                        key = cols[i].text.strip()
                        value = cols[i+1].text.strip()
                        data[key] = value
        
        return data
    except Exception:
        return {}

@st.cache_data(ttl=1800)
def get_historical_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Obtiene precios históricos con rate limiting"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            time.sleep(1)
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if not hist.empty:
                return hist
            
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"⚠️ No se pudo obtener histórico de {ticker}")
                return pd.DataFrame()
            time.sleep(2 * (attempt + 1))
    
    return pd.DataFrame()

# =========================
# FUNCIONES DE ANÁLISIS
# =========================

def calculate_metrics(prices: pd.DataFrame) -> Dict:
    """Calcula métricas financieras"""
    if prices.empty:
        return {}
    
    returns = prices['Close'].pct_change().dropna()
    
    metrics = {
        'rendimiento_total': ((prices['Close'].iloc[-1] / prices['Close'].iloc[0]) - 1) * 100,
        'volatilidad_anual': returns.std() * np.sqrt(252) * 100,
        'rendimiento_anual': returns.mean() * 252 * 100,
        'max_drawdown': ((prices['Close'] / prices['Close'].cummax()) - 1).min() * 100,
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
    }
    
    return metrics

def compare_stocks(tickers: List[str], progress_bar=None) -> pd.DataFrame:
    """Compara múltiples acciones con progress bar"""
    comparison_data = []
    total = len(tickers)
    
    for idx, ticker in enumerate(tickers):
        if progress_bar:
            progress_bar.progress((idx + 1) / total, text=f"Analizando {ticker}...")
        
        yf_data = scrape_yfinance_data(ticker)
        hist = get_historical_prices(ticker, "1y")
        metrics = calculate_metrics(hist)
        
        if yf_data:
            comparison_data.append({
                'Ticker': ticker,
                'Nombre': yf_data.get('nombre', 'N/D'),
                'Sector': yf_data.get('sector', 'N/D'),
                'Precio': f"${yf_data.get('precio_actual', 0):.2f}",
                'Market Cap': f"${yf_data.get('market_cap', 0)/1e9:.2f}B" if yf_data.get('market_cap', 0) > 0 else "N/D",
                'P/E': f"{yf_data.get('pe_ratio', 0):.2f}" if yf_data.get('pe_ratio', 0) else "N/D",
                'Beta': f"{yf_data.get('beta', 0):.2f}" if yf_data.get('beta', 0) else "N/D",
                'Div. Yield': f"{yf_data.get('dividend_yield', 0)*100:.2f}%" if yf_data.get('dividend_yield', 0) else "0.00%",
                'Rendimiento 1Y': f"{metrics.get('rendimiento_total', 0):.2f}%",
                'Volatilidad': f"{metrics.get('volatilidad_anual', 0):.2f}%",
                'Sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}"
            })
        else:
            st.warning(f"⚠️ No se pudieron obtener datos de {ticker}")
    
    return pd.DataFrame(comparison_data)

# =========================
# FUNCIÓN DE IA
# =========================

def analyze_with_gemini(portfolio_data: pd.DataFrame, profile: str) -> str:
    """Análisis con Gemini"""
    perfiles = {
        "Conservador": "bajo riesgo, enfocado en preservación de capital y dividendos estables",
        "Moderado": "riesgo medio, balance entre crecimiento y estabilidad",
        "Agresivo": "alto riesgo, enfocado en máximo crecimiento a largo plazo"
    }
    
    perfil_descripcion = perfiles.get(profile, perfiles["Moderado"])
    
    prompt = f"""
    Eres un asesor financiero experto. Analiza el siguiente portafolio de acciones para un inversionista con perfil {profile} ({perfil_descripcion}).
    
    Datos del portafolio:
    {portfolio_data.to_string()}
    
    Proporciona un análisis detallado que incluya:
    
    1. **Evaluación General del Portafolio**
       - Diversificación sectorial
       - Balance riesgo-retorno
    
    2. **Recomendaciones Específicas por Acción**
       - Qué acciones mantener, comprar más o vender
       - Justificación basada en métricas
    
    3. **Estrategia para el Perfil {profile}**
       - Ajustes recomendados
       - Ponderación sugerida
    
    4. **Conclusión y Acción Inmediata**
    
    Sé específico y técnico.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generando análisis: {str(e)}"

# =========================
# INTERFAZ
# =========================

st.markdown("""
<div class="main-header">
    <h1>📊 Análisis Comparativo de Portafolio</h1>
    <p>Análisis avanzado de acciones con datos de Yahoo Finance, Finviz e IA</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("Selección de Acciones")
    ticker_input = st.text_area(
        "Ingresa los tickers (uno por línea)",
        value="AAPL\nMSFT\nGOOGL",
        height=120,
        help="⚠️ Máximo 5 acciones para evitar rate limiting"
    )
    
    tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]
    
    # Limitar a 5 acciones
    if len(tickers) > 5:
        st.warning("⚠️ Limitado a 5 acciones para evitar rate limiting")
        tickers = tickers[:5]
    
    st.metric("Acciones Seleccionadas", len(tickers))
    
    st.markdown("---")
    
    st.subheader("🎯 Perfil de Inversión")
    perfil = st.selectbox(
        "Selecciona tu perfil",
        ["Conservador", "Moderado", "Agresivo"],
        index=1
    )
    
    st.markdown("---")
    
    st.subheader("📅 Período")
    periodo = st.selectbox(
        "Período histórico",
        ["1mo", "3mo", "6mo", "1y"],
        index=3
    )
    
    st.markdown("---")
    
    analyze_btn = st.button("🚀 ANALIZAR PORTAFOLIO", type="primary", use_container_width=True)

if analyze_btn:
    if not tickers:
        st.error("❌ Por favor ingresa al menos un ticker")
        st.stop()
    
    st.info("⏳ Obteniendo datos... Esto puede tomar unos segundos para evitar rate limiting")
    
    progress_bar = st.progress(0, text="Iniciando análisis...")
    
    try:
        st.header("📊 Comparación de Acciones")
        comparison_df = compare_stocks(tickers, progress_bar)
        
        progress_bar.empty()
        
        if not comparison_df.empty:
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            csv = comparison_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar CSV",
                csv,
                f"portafolio_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
            
            st.markdown("---")
            
            # Gráficos
            st.header("📈 Desempeño Histórico")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_prices = go.Figure()
                
                for ticker in tickers:
                    hist = get_historical_prices(ticker, periodo)
                    if not hist.empty:
                        normalized = (hist['Close'] / hist['Close'].iloc[0]) * 100
                        fig_prices.add_trace(go.Scatter(
                            x=hist.index,
                            y=normalized,
                            name=ticker,
                            mode='lines'
                        ))
                
                fig_prices.update_layout(
                    title="Precios Normalizados (Base 100)",
                    xaxis_title="Fecha",
                    yaxis_title="Valor",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_prices, use_container_width=True)
            
            with col2:
                volatilidades = []
                for ticker in tickers:
                    hist = get_historical_prices(ticker, periodo)
                    metrics = calculate_metrics(hist)
                    volatilidades.append({
                        'Ticker': ticker,
                        'Volatilidad': metrics.get('volatilidad_anual', 0)
                    })
                
                vol_df = pd.DataFrame(volatilidades)
                
                fig_vol = go.Figure(data=[
                    go.Bar(
                        x=vol_df['Ticker'],
                        y=vol_df['Volatilidad'],
                        marker_color='lightblue',
                        text=vol_df['Volatilidad'].apply(lambda x: f'{x:.1f}%'),
                        textposition='outside'
                    )
                ])
                
                fig_vol.update_layout(
                    title="Volatilidad Anual",
                    xaxis_title="Ticker",
                    yaxis_title="Volatilidad (%)",
                    height=400
                )
                st.plotly_chart(fig_vol, use_container_width=True)
            
            st.markdown("---")
            
            # Análisis IA
            st.markdown("""
            <div class="ai-section">
                <h2>🤖 Análisis con IA</h2>
                <p>Análisis personalizado según tu perfil</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("🧠 Generando análisis..."):
                ai_analysis = analyze_with_gemini(comparison_df, perfil)
                st.markdown(ai_analysis)
            
            st.success("✅ Análisis completado")
        else:
            st.error("❌ No se pudieron obtener datos. Intenta de nuevo en unos minutos.")
    
    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👈 Configura tu portafolio y presiona ANALIZAR")
    
    st.warning("⚠️ **Importante**: Limita el análisis a 5 acciones máximo para evitar rate limiting de Yahoo Finance")

st.markdown("---")
st.caption("📊 Análisis de Portafolio | Yahoo Finance & Finviz | Gemini AI")