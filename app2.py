from __future__ import annotations
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import os
import time
import json

warnings.filterwarnings("ignore")

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(
    page_title="Asesor de Portafolio con IA",
    layout="wide",
    page_icon="🤖"
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

# Inicializar session state
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None

# =========================
# ESTILOS CSS MEJORADOS
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -1px;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    .questionnaire-card {
        background: white;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid #e0e0e0;
    }
    
    .ai-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 12px 30px rgba(245, 87, 108, 0.3);
    }
    
    .ai-section h2 {
        margin: 0 0 0.5rem 0;
        font-size: 2rem;
        font-weight: 800;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    .metric-sublabel {
        color: #adb5bd;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }
    
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 0.75rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    .kpi-card h4 {
        margin: 0 0 0.5rem 0;
        color: #667eea;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-card .value {
        font-size: 2rem;
        font-weight: 900;
        color: #333;
        margin: 0.5rem 0;
    }
    
    .kpi-card .description {
        font-size: 0.85rem;
        color: #6c757d;
        line-height: 1.4;
    }
    
    .stock-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
    }
    
    .stock-header h2 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 900;
    }
    
    .stock-header .ticker {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.25rem;
    }
    
    .back-button {
        background: white;
        color: #667eea;
        border: 2px solid #667eea;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    .back-button:hover {
        background: #667eea;
        color: white;
        transform: translateX(-5px);
    }
    
    .enhanced-table {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
    }
    
    .stDataFrame th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 1rem !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    .stDataFrame td {
        padding: 1rem !important;
        border-bottom: 1px solid #f0f0f0 !important;
    }
    
    .stDataFrame tr:hover {
        background: #f8f9fa !important;
    }
    
    .success-badge {
        display: inline-block;
        background: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    .click-instruction {
        background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        color: #3f51b5;
        margin: 1rem 0;
        border: 2px dashed #3f51b5;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)

# =========================
# MAPEO DE BOLSAS
# =========================
BOLSAS_SUFIJOS = {
    "NYSE/NASDAQ (USA)": "",
    "Bolsa Mexicana de Valores (BMV)": ".MX",
    "Bolsa de Toronto (TSX)": ".TO",
    "Bolsa de Londres (LSE)": ".L",
    "Bolsa de Frankfurt (FRA)": ".DE",
    "Bolsa de París (EPA)": ".PA",
    "Bolsa de Madrid (BME)": ".MC",
    "Bolsa de São Paulo (B3)": ".SA",
    "Bolsa de Buenos Aires (BCBA)": ".BA",
    "Bolsa de Tokio (TSE)": ".T",
    "Bolsa de Hong Kong (HKEX)": ".HK",
    "Bolsa de Shanghái (SSE)": ".SS",
    "Bolsa de Australia (ASX)": ".AX"
}

# =========================
# FUNCIONES DE DATOS
# =========================

@st.cache_data(ttl=3600)
def get_company_details(ticker: str) -> Dict:
    """Obtiene detalles completos de la empresa"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Obtener datos financieros
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        
        return {
            'nombre': info.get('longName', info.get('shortName', ticker)),
            'sector': info.get('sector', 'N/D'),
            'industria': info.get('industry', 'N/D'),
            'empleados': info.get('fullTimeEmployees', 0),
            'descripcion': info.get('longBusinessSummary', 'N/D'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'price_to_book': info.get('priceToBook', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'week_52_high': info.get('fiftyTwoWeekHigh', 0),
            'week_52_low': info.get('fiftyTwoWeekLow', 0),
            'avg_volume': info.get('averageVolume', 0),
            'profit_margin': info.get('profitMargins', 0),
            'operating_margin': info.get('operatingMargins', 0),
            'roe': info.get('returnOnEquity', 0),
            'roa': info.get('returnOnAssets', 0),
            'revenue_growth': info.get('revenueGrowth', 0),
            'earnings_growth': info.get('earningsGrowth', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'current_ratio': info.get('currentRatio', 0),
            'free_cashflow': info.get('freeCashflow', 0),
            'website': info.get('website', ''),
            'country': info.get('country', 'N/D'),
            'city': info.get('city', 'N/D')
        }
    except Exception as e:
        return {
            'nombre': ticker,
            'sector': 'N/D',
            'industria': 'N/D',
            'empleados': 0,
            'descripcion': 'No disponible',
            'market_cap': 0,
            'pe_ratio': 0,
            'beta': 0
        }

@st.cache_data(ttl=3600)
def get_batch_stock_data(tickers: List[str], period: str = "1y") -> Dict:
    """Descarga datos en batch"""
    try:
        tickers_str = " ".join(tickers)
        data = yf.download(
            tickers_str,
            period=period,
            group_by='ticker',
            auto_adjust=True,
            progress=False,
            threads=False
        )
        
        result = {}
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    ticker_data = data
                else:
                    ticker_data = data[ticker] if ticker in data.columns.get_level_values(0) else pd.DataFrame()
                
                if not ticker_data.empty and 'Close' in ticker_data.columns:
                    result[ticker] = {
                        'history': ticker_data,
                        'current_price': float(ticker_data['Close'].iloc[-1]),
                        'success': True
                    }
                else:
                    result[ticker] = {'success': False, 'history': pd.DataFrame()}
            except:
                result[ticker] = {'success': False, 'history': pd.DataFrame()}
        
        return result
    except Exception as e:
        return {}

def calculate_metrics(prices: pd.DataFrame) -> Dict:
    """Calcula métricas financieras"""
    if prices.empty or 'Close' not in prices.columns:
        return {}
    
    returns = prices['Close'].pct_change().dropna()
    
    metrics = {
        'rendimiento_total': ((prices['Close'].iloc[-1] / prices['Close'].iloc[0]) - 1) * 100,
        'volatilidad_anual': returns.std() * np.sqrt(252) * 100,
        'rendimiento_anual': returns.mean() * 252 * 100,
        'max_drawdown': ((prices['Close'] / prices['Close'].cummax()) - 1).min() * 100,
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
        'returns': returns
    }
    
    return metrics

def calculate_portfolio_metrics(batch_data: Dict, weights: Dict) -> Dict:
    """Calcula métricas del portafolio ponderado"""
    portfolio_returns = None
    dates = None
    
    for ticker, weight in weights.items():
        if ticker in batch_data and batch_data[ticker]['success']:
            hist = batch_data[ticker]['history']
            if not hist.empty and 'Close' in hist.columns:
                returns = hist['Close'].pct_change().dropna()
                
                if portfolio_returns is None:
                    portfolio_returns = returns * weight
                    dates = returns.index
                else:
                    portfolio_returns = portfolio_returns.add(returns * weight, fill_value=0)
    
    if portfolio_returns is None or portfolio_returns.empty:
        return {}
    
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    metrics = {
        'rendimiento_total': (cumulative_returns.iloc[-1] - 1) * 100,
        'volatilidad_anual': portfolio_returns.std() * np.sqrt(252) * 100,
        'rendimiento_anual': portfolio_returns.mean() * 252 * 100,
        'max_drawdown': ((cumulative_returns / cumulative_returns.cummax()) - 1).min() * 100,
        'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() != 0 else 0,
        'cumulative_returns': cumulative_returns,
        'dates': dates
    }
    
    return metrics

# =========================
# FUNCIONES DE IA
# =========================

def generate_portfolio_with_gemini(perfil: str, horizonte: str, capital: str, objetivos: List[str], sector_pref: str, bolsas: List[str]) -> Tuple[List[Dict], str]:
    """Genera un portafolio sugerido con Gemini"""
    
    objetivos_str = ", ".join(objetivos)
    bolsas_str = ", ".join(bolsas)
    
    # Construir sufijos para las bolsas seleccionadas
    sufijos_permitidos = []
    for bolsa_nombre in bolsas:
        sufijo = BOLSAS_SUFIJOS.get(bolsa_nombre, "")
        if sufijo:
            sufijos_permitidos.append(sufijo)
        else:
            sufijos_permitidos.append("")  # NYSE/NASDAQ no tiene sufijo
    
    prompt = f"""
    Eres un asesor financiero experto. Basándote en el siguiente perfil de inversionista, sugiere un portafolio óptimo de inversión.
    
    **Perfil del Inversionista:**
    - Perfil de riesgo: {perfil}
    - Horizonte temporal: {horizonte}
    - Capital aproximado: {capital}
    - Objetivos: {objetivos_str}
    - Preferencia sectorial: {sector_pref}
    - Bolsas de preferencia: {bolsas_str}
    
    **Instrucciones:**
    1. Sugiere entre 5 y 10 acciones/ETFs de las bolsas seleccionadas: {bolsas_str}
    2. Para NYSE/NASDAQ: usa tickers sin sufijo (ej: AAPL, MSFT)
    3. Para BMV: usa sufijo .MX (ej: WALMEX.MX, GMEXICOB.MX)
    4. Para otras bolsas: usa el sufijo correcto
    5. Perfil Conservador: ETFs diversificados, blue-chips, dividendos
    6. Perfil Moderado: balance crecimiento/estabilidad
    7. Perfil Agresivo: alto crecimiento, tecnología
    8. Diversifica por sectores y geografía
    
    **Responde ÚNICAMENTE con un JSON válido:**
```json
    {{
        "portafolio": [
            {{"ticker": "AAPL", "peso": 15, "razon": "Líder tecnológico con fundamentales sólidos"}},
            {{"ticker": "WALMEX.MX", "peso": 10, "razon": "Retail líder en México"}},
            ...
        ],
        "justificacion": "Este portafolio..."
    }}
```
    
    **IMPORTANTE:** 
    - Pesos deben sumar 100
    - Solo tickers válidos y existentes
    - 5-10 posiciones
    - Solo JSON, sin texto adicional
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        portfolio_data = json.loads(response_text)
        portafolio = portfolio_data.get('portafolio', [])
        justificacion = portfolio_data.get('justificacion', '')
        
        # Normalizar pesos
        total_peso = sum([p['peso'] for p in portafolio])
        if abs(total_peso - 100) > 1:
            for p in portafolio:
                p['peso'] = (p['peso'] / total_peso) * 100
        
        return portafolio, justificacion
        
    except Exception as e:
        st.error(f"Error generando portafolio: {str(e)}")
        return [], ""

def analyze_portfolio_with_gemini(portfolio_data: pd.DataFrame, metrics: Dict, perfil: str, justificacion_inicial: str) -> str:
    """Análisis final del portafolio - MÁXIMO 300 PALABRAS"""
    
    prompt = f"""
    Análisis CONCISO del portafolio (MÁXIMO 300 PALABRAS).
    
    **Portafolio Inicial:** {justificacion_inicial}
    
    **Datos:** {portfolio_data.to_string()}
    
    **Métricas:**
    - Rendimiento Total: {metrics.get('rendimiento_total', 0):.2f}%
    - Rendimiento Anual: {metrics.get('rendimiento_anual', 0):.2f}%
    - Volatilidad: {metrics.get('volatilidad_anual', 0):.2f}%
    - Sharpe: {metrics.get('sharpe_ratio', 0):.2f}
    - Max DD: {metrics.get('max_drawdown', 0):.2f}%
    
    **Perfil:** {perfil}
    
    Incluye (300 palabras MAX):
    1. Evaluación desempeño (50 palabras)
    2. Top posiciones (80 palabras)
    3. Gestión riesgo (70 palabras)
    4. Recomendaciones (100 palabras)
    
    Sé directo y específico.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# DASHBOARD DE ACCIÓN INDIVIDUAL
# =========================

def render_stock_dashboard(ticker: str, peso: float, razon: str, batch_data: Dict):
    """Renderiza dashboard detallado de una acción específica"""
    
    # Botón para volver
    if st.button("← Volver al Portafolio", key="back_button"):
        st.session_state.selected_stock = None
        st.rerun()
    
    # Obtener detalles completos
    with st.spinner(f"📊 Cargando información detallada de {ticker}..."):
        details = get_company_details(ticker)
        time.sleep(0.5)
    
    # Header de la acción
    st.markdown(f"""
    <div class="stock-header">
        <h2>{details['nombre']}</h2>
        <div class="ticker">{ticker} | {details['sector']} | {details['industria']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Información básica
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <h4>📍 Ubicación</h4>
            <div class="value">{details['city']}</div>
            <div class="description">{details['country']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        empleados = f"{details['empleados']:,}" if details['empleados'] > 0 else "N/D"
        st.markdown(f"""
        <div class="kpi-card">
            <h4>👥 Empleados</h4>
            <div class="value">{empleados}</div>
            <div class="description">Fuerza laboral</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        market_cap = details['market_cap']
        if market_cap > 1e12:
            cap_str = f"${market_cap/1e12:.2f}T"
        elif market_cap > 1e9:
            cap_str = f"${market_cap/1e9:.2f}B"
        elif market_cap > 1e6:
            cap_str = f"${market_cap/1e6:.2f}M"
        else:
            cap_str = "N/D"
        
        st.markdown(f"""
        <div class="kpi-card">
            <h4>💰 Market Cap</h4>
            <div class="value">{cap_str}</div>
            <div class="description">Capitalización de mercado</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <h4>📊 Peso en Portafolio</h4>
            <div class="value" style="color: #667eea">{peso:.1f}%</div>
            <div class="description">Ponderación asignada</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Razón de inclusión
    st.markdown(f"""
    <div class="info-card">
        <strong>💡 Razón de inclusión en el portafolio:</strong><br>
        {razon}
    </div>
    """, unsafe_allow_html=True)
    
    # Descripción de la empresa
    if details['descripcion'] != 'N/D':
        with st.expander("📄 Sobre la empresa"):
            st.write(details['descripcion'])
            if details['website']:
                st.markdown(f"🌐 **Website:** [{details['website']}]({details['website']})")
    
    st.markdown("---")
    
    # KPIs Bursátiles y Corporativos
    st.header("📊 KPIs Bursátiles y Corporativos")
    
    tab1, tab2, tab3 = st.tabs(["📈 Métricas de Valuación", "💼 Métricas Operativas", "📊 Análisis Técnico"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pe = details['pe_ratio']
            pe_color = "#4CAF50" if 0 < pe < 20 else "#FF9800" if 0 < pe < 30 else "#f44336" if pe > 30 else "#999"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">P/E Ratio</div>
                <div class="metric-value" style="color: {pe_color}">{pe:.2f if pe else 'N/D'}</div>
                <div class="metric-sublabel">Precio/Ganancias</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            forward_pe = details['forward_pe']
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Forward P/E</div>
                <div class="metric-value">{forward_pe:.2f if forward_pe else 'N/D'}</div>
                <div class="metric-sublabel">P/E Proyectado</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            pb = details['price_to_book']
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">P/B Ratio</div>
                <div class="metric-value">{pb:.2f if pb else 'N/D'}</div>
                <div class="metric-sublabel">Precio/Valor Libro</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            peg = details['peg_ratio']
            peg_color = "#4CAF50" if 0 < peg < 1 else "#FF9800" if 0 < peg < 2 else "#f44336" if peg > 2 else "#999"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">PEG Ratio</div>
                <div class="metric-value" style="color: {peg_color}">{peg:.2f if peg else 'N/D'}</div>
                <div class="metric-sublabel">P/E / Crecimiento</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            div_yield = details['dividend_yield'] * 100 if details['dividend_yield'] else 0
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #4CAF50">
                <h4>💵 Dividend Yield</h4>
                <div class="value" style="color: #4CAF50">{div_yield:.2f}%</div>
                <div class="description">Rendimiento por dividendos</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            beta = details['beta']
            beta_color = "#4CAF50" if beta < 1 else "#FF9800" if beta < 1.5 else "#f44336"
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: {beta_color}">
                <h4>📊 Beta</h4>
                <div class="value" style="color: {beta_color}">{beta:.2f if beta else 'N/D'}</div>
                <div class="description">Volatilidad vs mercado</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            w52_high = details['week_52_high']
            w52_low = details['week_52_low']
            if ticker in batch_data and batch_data[ticker]['success']:
                current = batch_data[ticker]['current_price']
                if w52_high and w52_low:
                    pct_from_high = ((current - w52_high) / w52_high) * 100
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left-color: #2196F3">
                        <h4>📏 Rango 52 Semanas</h4>
                        <div class="value" style="font-size: 1.3rem; color: #2196F3">${w52_low:.2f} - ${w52_high:.2f}</div>
                        <div class="description">Actual: ${current:.2f} ({pct_from_high:+.1f}% desde máximo)</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            profit_margin = details['profit_margin'] * 100 if details['profit_margin'] else 0
            pm_color = "#4CAF50" if profit_margin > 20 else "#FF9800" if profit_margin > 10 else "#f44336"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Margen de Utilidad</div>
                <div class="metric-value" style="color: {pm_color}">{profit_margin:.1f}%</div>
                <div class="metric-sublabel">Profit Margin</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            op_margin = details['operating_margin'] * 100 if details['operating_margin'] else 0
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Margen Operativo</div>
                <div class="metric-value">{op_margin:.1f}%</div>
                <div class="metric-sublabel">Operating Margin</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            roe = details['roe'] * 100 if details['roe'] else 0
            roe_color = "#4CAF50" if roe > 15 else "#FF9800" if roe > 10 else "#f44336"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">ROE</div>
                <div class="metric-value" style="color: {roe_color}">{roe:.1f}%</div>
                <div class="metric-sublabel">Return on Equity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            roa = details['roa'] * 100 if details['roa'] else 0
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">ROA</div>
                <div class="metric-value">{roa:.1f}%</div>
                <div class="metric-sublabel">Return on Assets</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rev_growth = details['revenue_growth'] * 100 if details['revenue_growth'] else 0
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #4CAF50">
                <h4>📈 Crecimiento de Ingresos</h4>
                <div class="value" style="color: {'#4CAF50' if rev_growth > 0 else '#f44336'}">{rev_growth:+.1f}%</div>
                <div class="description">Revenue Growth (YoY)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            earnings_growth = details['earnings_growth'] * 100 if details['earnings_growth'] else 0
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #FF9800">
                <h4>💰 Crecimiento de Ganancias</h4>
                <div class="value" style="color: {'#4CAF50' if earnings_growth > 0 else '#f44336'}">{earnings_growth:+.1f}%</div>
                <div class="description">Earnings Growth (YoY)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            de_ratio = details['debt_to_equity']
            de_color = "#4CAF50" if de_ratio < 0.5 else "#FF9800" if de_ratio < 1.5 else "#f44336"
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: {de_color}">
                <h4>📊 Deuda/Capital</h4>
                <div class="value" style="color: {de_color}">{de_ratio:.2f if de_ratio else 'N/D'}</div>
                <div class="description">Debt-to-Equity Ratio</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        if ticker in batch_data and batch_data[ticker]['success']:
            hist = batch_data[ticker]['history']
            metrics = calculate_metrics(hist)
            
            # Gráfico de precio
            fig_price = go.Figure()
            
            fig_price.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Precio'
            ))
            
            # Medias móviles
            sma_20 = hist['Close'].rolling(window=20).mean()
            sma_50 = hist['Close'].rolling(window=50).mean()
            
            fig_price.add_trace(go.Scatter(
                x=hist.index,
                y=sma_20,
                name='SMA 20',
                line=dict(color='#FF9800', width=2)
            ))
            
            fig_price.add_trace(go.Scatter(
                x=hist.index,
                y=sma_50,
                name='SMA 50',
                line=dict(color='#2196F3', width=2)
            ))
            
            fig_price.update_layout(
                title=f'Gráfico de Velas - {ticker}',
                yaxis_title='Precio ($)',
                xaxis_rangeslider_visible=False,
                height=500,
                template='plotly_white',
                font=dict(family='Inter')
            )
            
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Métricas técnicas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rend = metrics.get('rendimiento_total', 0)
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Rendimiento</div>
                    <div class="metric-value" style="color: {'#4CAF50' if rend > 0 else '#f44336'}">{rend:+.2f}%</div>
                    <div class="metric-sublabel">1 Año</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                vol = metrics.get('volatilidad_anual', 0)
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Volatilidad</div>
                    <div class="metric-value">{vol:.1f}%</div>
                    <div class="metric-sublabel">Anualizada</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                sharpe = metrics.get('sharpe_ratio', 0)
                sharpe_color = "#4CAF50" if sharpe > 1 else "#FF9800" if sharpe > 0 else "#f44336"
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value" style="color: {sharpe_color}">{sharpe:.2f}</div>
                    <div class="metric-sublabel">Riesgo/Retorno</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                dd = metrics.get('max_drawdown', 0)
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value" style="color: #f44336">{dd:.1f}%</div>
                    <div class="metric-sublabel">Pérdida Máxima</div>
                </div>
                """, unsafe_allow_html=True)

# =========================
# INTERFAZ PRINCIPAL
# =========================

# Verificar si hay una acción seleccionada
if st.session_state.selected_stock:
    # Renderizar dashboard de la acción
    stock_info = st.session_state.selected_stock
    render_stock_dashboard(
        stock_info['ticker'],
        stock_info['peso'],
        stock_info['razon'],
        st.session_state.portfolio_data['batch_data']
    )
    st.stop()

# Si no hay acción seleccionada, mostrar flujo normal
st.markdown("""
<div class="main-header">
    <h1>🤖 Asesor de Portafolio con IA</h1>
    <p>Portafolios personalizados generados por Gemini 2.5 Flash</p>
</div>
""", unsafe_allow_html=True)

# CUESTIONARIO
st.markdown('<div class="questionnaire-card">', unsafe_allow_html=True)
st.header("📋 Cuestionario de Perfil de Inversión")

col1, col2 = st.columns(2)

with col1:
    perfil = st.selectbox(
        "🎯 ¿Cuál es tu perfil de riesgo?",
        ["Conservador", "Moderado", "Agresivo"]
    )
    
    horizonte = st.selectbox(
        "⏰ ¿Cuál es tu horizonte de inversión?",
        ["Corto plazo (< 1 año)", "Mediano plazo (1-5 años)", "Largo plazo (> 5 años)"],
        index=1
    )
    
    capital = st.selectbox(
        "💰 ¿Cuál es tu capital aproximado de inversión?",
        ["< $10,000", "$10,000 - $50,000", "$50,000 - $100,000", "> $100,000"],
        index=1
    )

with col2:
    objetivos = st.multiselect(
        "🎯 ¿Cuáles son tus objetivos de inversión?",
        [
            "Crecimiento de capital",
            "Generación de ingresos (dividendos)",
            "Preservación de capital",
            "Diversificación internacional",
            "Inversión ESG/Sostenible"
        ],
        default=["Crecimiento de capital"]
    )
    
    sector_pref = st.selectbox(
        "🏢 ¿Preferencia sectorial?",
        ["Sin preferencia (diversificado)", "Tecnología", "Salud", "Finanzas", "Energía", "Consumo"],
        index=0
    )
    
    # NUEVA PREGUNTA: Bolsas de valores
    bolsas_preferidas = st.multiselect(
        "🌍 ¿En qué bolsas te gustaría invertir?",
        list(BOLSAS_SUFIJOS.keys()),
        default=["NYSE/NASDAQ (USA)"],
        help="Selecciona una o más bolsas de valores"
    )

st.markdown('</div>', unsafe_allow_html=True)

generate_btn = st.button("🚀 GENERAR PORTAFOLIO CON IA", type="primary", use_container_width=True)

# GENERACIÓN Y ANÁLISIS
if generate_btn:
    
    if not objetivos:
        st.error("❌ Selecciona al menos un objetivo")
        st.stop()
    
    if not bolsas_preferidas:
        st.error("❌ Selecciona al menos una bolsa de valores")
        st.stop()
    
    # Generar portafolio
    with st.spinner("🧠 IA generando portafolio..."):
        portafolio, justificacion = generate_portfolio_with_gemini(
            perfil, horizonte, capital, objetivos, sector_pref, bolsas_preferidas
        )
    
    if not portafolio:
        st.error("❌ No se pudo generar el portafolio")
        st.stop()
    
    st.markdown(f'<div class="success-badge">✅ Portafolio de {len(portafolio)} posiciones generado</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Obtener detalles
    with st.spinner("📥 Obteniendo detalles..."):
        portfolio_details = []
        for item in portafolio:
            details = get_company_details(item['ticker'])
            portfolio_details.append({**item, **details})
        time.sleep(1)
    
    # Gráfico de pastel MEJORADO
    st.header("📊 Portafolio Sugerido por IA")
    st.markdown(f'<div class="info-card"><strong>💡 Estrategia:</strong> {justificacion}</div>', unsafe_allow_html=True)
    
    # Instrucción de click
    st.markdown("""
    <div class="click-instruction">
        👆 Haz clic en cualquier porción del gráfico para ver el análisis detallado de esa acción
    </div>
    """, unsafe_allow_html=True)
    
    # Crear pie chart con mejor hover y clickeable
    hover_text = []
    for item in portfolio_details:
        empleados_str = f"{item['empleados']:,}" if item['empleados'] > 0 else "N/D"
        hover_info = (
            f"<b style='font-size:16px'>{item['nombre']}</b><br><br>"
            f"<b>Ticker:</b> {item['ticker']}<br>"
            f"<b>Sector:</b> {item['sector']}<br>"
            f"<b>Industria:</b> {item['industria']}<br>"
            f"<b>Empleados:</b> {empleados_str}<br>"
            f"<b>Ponderación:</b> {item['peso']:.1f}%<br><br>"
            f"<b>💡 Razón de selección:</b><br>"
            f"{item['razon']}<br><br>"
            f"<i>👆 Click para ver dashboard completo</i>"
        )
        hover_text.append(hover_info)
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']
    
    fig_pie = go.Figure()
    
    fig_pie.add_trace(go.Pie(
        labels=[f"<b>{item['ticker']}</b><br>{item['peso']:.1f}%" for item in portfolio_details],
        values=[item['peso'] for item in portfolio_details],
        text=[item['ticker'] for item in portfolio_details],
        textposition='inside',
        textfont=dict(size=16, color='white', family='Inter', weight='bold'),
        customdata=[[i, item['ticker'], item['peso'], item['razon']] for i, item in enumerate(portfolio_details)],
        hovertext=hover_text,
        hoverinfo='text',
        hole=0.45,
        marker=dict(
            colors=colors[:len(portfolio_details)],
            line=dict(color='white', width=4)
        ),
        pull=[0.08 if i == 0 else 0.02 for i in range(len(portfolio_details))]
    ))
    
    fig_pie.update_layout(
        title={
            'text': '<b>Composición del Portafolio</b><br><sub style="font-size:14px">Hover para detalles | Click para dashboard completo</sub>',
            'y':0.97,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=26, family='Inter', color='#333')
        },
        height=650,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=13, family='Inter'),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#667eea",
            borderwidth=2
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Inter",
            bordercolor="#667eea",
            align="left"
        ),
        margin=dict(l=20, r=250, t=100, b=20)
    )
    
    # Renderizar con event handling
    selected_points = st.plotly_chart(fig_pie, use_container_width=True, on_select="rerun", key="pie_chart")
    
    # Detectar clicks (workaround usando botones)
    st.markdown("### O selecciona directamente:")
    cols = st.columns(len(portfolio_details))
    for idx, (col, item) in enumerate(zip(cols, portfolio_details)):
        with col:
            if st.button(f"📊 {item['ticker']}", key=f"btn_{item['ticker']}", use_container_width=True):
                st.session_state.selected_stock = {
                    'ticker': item['ticker'],
                    'peso': item['peso'],
                    'razon': item['razon']
                }
                st.session_state.portfolio_data = {'batch_data': {}}
                st.rerun()
    
    # Continuar con el análisis del portafolio...
    tickers = [p['ticker'] for p in portafolio]
    weights = {p['ticker']: p['peso'] / 100 for p in portafolio}
    
    with st.spinner(f"📈 Descargando datos de {len(tickers)} activos..."):
        batch_data = get_batch_stock_data(tickers, "1y")
        time.sleep(1)
    
    # Guardar batch_data en session_state para uso posterior
    st.session_state.portfolio_data = {'batch_data': batch_data}
    
    successful_tickers = [t for t in tickers if t in batch_data and batch_data[t]['success']]
    failed_tickers = [t for t in tickers if t not in successful_tickers]
    
    if failed_tickers:
        st.markdown(f'<div class="warning-card">⚠️ No se obtuvieron datos de: {", ".join(failed_tickers)}</div>', unsafe_allow_html=True)
    
    if not successful_tickers:
        st.error("❌ No se pudieron obtener datos")
        st.stop()
    
    total_successful_weight = sum([weights[t] for t in successful_tickers])
    adjusted_weights = {t: weights[t] / total_successful_weight for t in successful_tickers}
    
    st.markdown(f'<div class="success-badge">✅ Datos obtenidos para {len(successful_tickers)} activos</div>', unsafe_allow_html=True)
    
    # Análisis individual
    st.markdown("---")
    st.header("📊 Análisis Individual de Activos")
    
    individual_data = []
    for ticker in successful_tickers:
        hist = batch_data[ticker]['history']
        metrics = calculate_metrics(hist)
        
        rend_icon = "🟢" if metrics.get('rendimiento_total', 0) > 0 else "🔴"
        sharpe_icon = "⭐" if metrics.get('sharpe_ratio', 0) > 1 else "⚠️" if metrics.get('sharpe_ratio', 0) > 0 else "❌"
        
        individual_data.append({
            'Ticker': ticker,
            'Peso': f"{adjusted_weights[ticker]*100:.1f}%",
            'Precio': f"${batch_data[ticker]['current_price']:.2f}",
            'Rendimiento': f"{rend_icon} {metrics.get('rendimiento_total', 0):.2f}%",
            'Volatilidad': f"{metrics.get('volatilidad_anual', 0):.1f}%",
            'Sharpe': f"{sharpe_icon} {metrics.get('sharpe_ratio', 0):.2f}",
            'Max DD': f"{metrics.get('max_drawdown', 0):.1f}%"
        })
    
    individual_df = pd.DataFrame(individual_data)
    
    st.markdown('<div class="enhanced-table">', unsafe_allow_html=True)
    st.dataframe(individual_df, use_container_width=True, hide_index=True, height=400)
    st.markdown('</div>', unsafe_allow_html=True)
    
    csv = individual_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Descargar CSV",
        csv,
        f"portafolio_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )
    
    # Métricas del portafolio
    st.markdown("---")
    st.header("📈 Métricas del Portafolio Ponderado")
    
    portfolio_metrics = calculate_portfolio_metrics(batch_data, adjusted_weights)
    
    if portfolio_metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            rend_total = portfolio_metrics.get('rendimiento_total', 0)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Rendimiento Total</div>
                <div class="metric-value" style="color: {'#4CAF50' if rend_total > 0 else '#f44336'}">{rend_total:+.2f}%</div>
                <div class="metric-sublabel">1 Año</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            rend_anual = portfolio_metrics.get('rendimiento_anual', 0)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Rendimiento Anual</div>
                <div class="metric-value" style="color: {'#4CAF50' if rend_anual > 0 else '#f44336'}">{rend_anual:+.2f}%</div>
                <div class="metric-sublabel">Anualizado</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            vol = portfolio_metrics.get('volatilidad_anual', 0)
            vol_color = '#4CAF50' if vol < 15 else '#FF9800' if vol < 25 else '#f44336'
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Volatilidad</div>
                <div class="metric-value" style="color: {vol_color}">{vol:.1f}%</div>
                <div class="metric-sublabel">Anual</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            sharpe = portfolio_metrics.get('sharpe_ratio', 0)
            sharpe_color = '#4CAF50' if sharpe > 1.5 else '#FF9800' if sharpe > 0.5 else '#f44336'
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value" style="color: {sharpe_color}">{sharpe:.2f}</div>
                <div class="metric-sublabel">{'⭐⭐⭐' if sharpe > 1.5 else '⭐⭐' if sharpe > 0.5 else '⭐'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            dd = portfolio_metrics.get('max_drawdown', 0)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value" style="color: #f44336">{dd:.1f}%</div>
                <div class="metric-sublabel">Pérdida Máxima</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráficos (tabs anteriores)
        st.markdown("---")
        st.header("📊 Visualización del Portafolio")
        
        tab1, tab2 = st.tabs(["📈 Desempeño", "📊 Comparación"])
        
        with tab1:
            cumulative = portfolio_metrics.get('cumulative_returns')
            dates = portfolio_metrics.get('dates')
            
            if cumulative is not None:
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Scatter(
                    x=dates,
                    y=(cumulative - 1) * 100,
                    line=dict(color='#667eea', width=4),
                    fill='tonexty',
                    fillcolor='rgba(102, 126, 234, 0.2)'
                ))
                fig_perf.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_perf.update_layout(
                    title='Rendimiento Acumulado del Portafolio',
                    yaxis_title='Rendimiento (%)',
                    height=500,
                    template='plotly_white'
                )
                st.plotly_chart(fig_perf, use_container_width=True)
        
        with tab2:
            fig_comp = go.Figure()
            for ticker in successful_tickers:
                hist = batch_data[ticker]['history']
                if not hist.empty:
                    normalized = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
                    fig_comp.add_trace(go.Scatter(
                        x=hist.index,
                        y=normalized,
                        name=f"{ticker} ({adjusted_weights[ticker]*100:.1f}%)",
                        mode='lines'
                    ))
            
            if cumulative is not None:
                fig_comp.add_trace(go.Scatter(
                    x=dates,
                    y=(cumulative - 1) * 100,
                    name='PORTAFOLIO',
                    line=dict(color='#667eea', width=5)
                ))
            
            fig_comp.update_layout(
                title='Comparación Individual vs Portafolio',
                yaxis_title='Rendimiento (%)',
                height=550,
                template='plotly_white'
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
        # Análisis IA final
        st.markdown("---")
        st.markdown("""
        <div class="ai-section">
            <h2>🤖 Análisis Final con IA</h2>
            <p>Evaluación concisa del portafolio (máx. 300 palabras)</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🧠 Generando análisis..."):
            final_analysis = analyze_portfolio_with_gemini(
                individual_df,
                portfolio_metrics,
                perfil,
                justificacion
            )
            st.markdown(final_analysis)
        
        st.success("✅ Análisis completo")

else:
    st.info("👆 Completa el cuestionario y genera tu portafolio")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 IA Personalizada
        - Gemini 2.5 Flash
        - Portafolio óptimo
        - 5-10 posiciones
        - Multi-bolsa
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Análisis Completo
        - Métricas avanzadas
        - Visualizaciones
        - Dashboard por acción
        - KPIs detallados
        """)
    
    with col3:
        st.markdown("""
        ### 💡 Interactivo
        - Click en el pie chart
        - Dashboard detallado
        - KPIs bursátiles
        - Métricas corporativas
        """)

st.markdown("---")
st.caption("🤖 Powered by Gemini 2.5 Flash | Yahoo Finance | © 2025")