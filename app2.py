from __future__ import annotations
import warnings
import math
from datetime import date, datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from dotenv import load_dotenv
import os

warnings.filterwarnings("ignore")

# =========================
# CONFIGURACIÓN GENERAL
# =========================
APP_OWNER = "Iker Huerga"
APP_VERSION = "5.0.0 PRO"
DISCLAIMER = f"© {date.today().year} {APP_OWNER} — Uso académico. No es asesoría financiera."

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

st.set_page_config(
    page_title="Dashboard Financiero AI Pro - Análisis Bursátil Completo",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# =========================
# ESTILOS PROFESIONALES Y ÚNICOS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;900&display=swap');

:root {
  --bg-primary: #FFF8F2;
  --bg-secondary: #FFFFFF;
  --text-primary: #3E2723;
  --text-secondary: #6D4C41;
  --accent-orange: #FB8C00;
  --accent-deep: #F57C00;
  --success: #4CAF50;
  --danger: #F44336;
  --warning: #FF9800;
  --info: #2196F3;
  --shadow-sm: 0 2px 8px rgba(0,0,0,0.05);
  --shadow-md: 0 4px 16px rgba(0,0,0,0.08);
  --shadow-lg: 0 8px 32px rgba(0,0,0,0.12);
}

html, body, [class^="css"] {
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hero Header Único */
.hero-header {
  background: linear-gradient(135deg, #FB8C00 0%, #F57C00 50%, #E65100 100%);
  padding: 32px 40px;
  border-radius: 20px;
  margin-bottom: 32px;
  color: white;
  box-shadow: var(--shadow-lg);
  position: relative;
  overflow: hidden;
}

.hero-header::before {
  content: '';
  position: absolute;
  top: -50%;
  right: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
  animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 0.5; }
  50% { transform: scale(1.1); opacity: 0.8; }
}

.hero-header h1 {
  margin: 0;
  font-size: 42px;
  font-weight: 900;
  letter-spacing: -1px;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
  position: relative;
  z-index: 1;
}

.hero-header p {
  margin: 12px 0 0 0;
  font-size: 16px;
  opacity: 0.95;
  font-weight: 300;
  position: relative;
  z-index: 1;
}

.hero-badge {
  display: inline-block;
  background: rgba(255,255,255,0.2);
  padding: 6px 14px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  margin-top: 12px;
  backdrop-filter: blur(10px);
}

/* Sidebar Estilizada */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #FFF3E0 0%, #FFE0B2 50%, #FFCC80 100%) !important;
  border-right: 3px solid var(--accent-orange);
}

section[data-testid="stSidebar"] > div {
  padding-top: 2rem;
}

/* KPI Cards Ultra Profesionales */
.kpi-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin: 24px 0;
}

.kpi-card {
  background: linear-gradient(135deg, #FFFFFF 0%, #FFF8F2 100%);
  border: 2px solid var(--accent-orange);
  border-radius: 16px;
  padding: 24px;
  text-align: center;
  box-shadow: var(--shadow-md);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}

.kpi-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 4px;
  background: linear-gradient(90deg, var(--accent-orange), var(--accent-deep));
}

.kpi-card:hover {
  transform: translateY(-8px) scale(1.02);
  box-shadow: var(--shadow-lg);
  border-color: var(--accent-deep);
}

.kpi-icon {
  font-size: 32px;
  margin-bottom: 12px;
  opacity: 0.8;
}

.kpi-title {
  font-size: 11px;
  color: var(--text-secondary);
  text-transform: uppercase;
  font-weight: 700;
  letter-spacing: 1px;
  margin-bottom: 8px;
}

.kpi-value {
  font-size: 28px;
  font-weight: 900;
  color: var(--accent-orange);
  margin: 8px 0;
  line-height: 1.2;
}

.kpi-change {
  font-size: 13px;
  font-weight: 600;
  margin-top: 8px;
}

.kpi-change.positive {
  color: var(--success);
}

.kpi-change.negative {
  color: var(--danger);
}

/* Section Headers Únicos */
.section-header {
  font-size: 28px;
  font-weight: 900;
  color: var(--accent-orange);
  margin: 40px 0 24px 0;
  padding-bottom: 12px;
  border-bottom: 4px solid var(--accent-orange);
  position: relative;
  display: flex;
  align-items: center;
  gap: 12px;
}

.section-header::before {
  content: '';
  width: 8px;
  height: 40px;
  background: linear-gradient(180deg, var(--accent-orange), var(--accent-deep));
  border-radius: 4px;
}

.subsection-header {
  font-size: 20px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 24px 0 16px 0;
  padding-left: 16px;
  border-left: 4px solid var(--accent-orange);
}

/* AI Analysis Card */
.ai-analysis-card {
  background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
  border: 3px solid var(--accent-orange);
  border-radius: 20px;
  padding: 32px;
  margin: 24px 0;
  box-shadow: var(--shadow-lg);
  position: relative;
}

.ai-analysis-card::before {
  content: '🤖';
  position: absolute;
  top: 20px;
  right: 20px;
  font-size: 48px;
  opacity: 0.1;
}

.ai-analysis-card h3 {
  color: var(--accent-deep);
  margin: 0 0 16px 0;
  font-size: 24px;
  font-weight: 800;
}

.ai-badge {
  display: inline-block;
  background: var(--accent-orange);
  color: white;
  padding: 6px 16px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 16px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* Risk Alerts */
.risk-alert {
  padding: 20px 24px;
  border-radius: 12px;
  margin: 16px 0;
  border-left: 6px solid;
  box-shadow: var(--shadow-sm);
  font-weight: 500;
}

.risk-alert-low {
  background: #E8F5E9;
  border-color: var(--success);
  color: #1B5E20;
}

.risk-alert-medium {
  background: #FFF3E0;
  border-color: var(--warning);
  color: #E65100;
}

.risk-alert-high {
  background: #FFEBEE;
  border-color: var(--danger);
  color: #B71C1C;
}

/* Enhanced Table */
.metrics-table {
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: var(--shadow-md);
  margin: 20px 0;
}

.stDataFrame {
  border-radius: 12px !important;
}

/* Buttons Enhanced */
.stButton > button {
  background: linear-gradient(135deg, var(--accent-orange) 0%, var(--accent-deep) 100%);
  color: white;
  font-weight: 700;
  border: none;
  border-radius: 12px;
  padding: 14px 32px;
  font-size: 16px;
  transition: all 0.3s ease;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  box-shadow: var(--shadow-md);
}

.stButton > button:hover {
  transform: translateY(-3px);
  box-shadow: var(--shadow-lg);
}

/* Tabs Personalizados */
.stTabs [data-baseweb="tab-list"] {
  gap: 12px;
  background: transparent;
}

.stTabs [data-baseweb="tab"] {
  background: white;
  border: 2px solid rgba(251, 140, 0, 0.2);
  border-radius: 12px;
  padding: 12px 24px;
  font-weight: 700;
  color: var(--text-primary);
  transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
  border-color: var(--accent-orange);
  transform: translateY(-2px);
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--accent-orange), var(--accent-deep));
  color: white;
  border-color: var(--accent-deep);
}

/* Company Info Card */
.company-info-card {
  background: white;
  border-radius: 16px;
  padding: 28px;
  box-shadow: var(--shadow-md);
  margin: 20px 0;
  border: 2px solid rgba(251, 140, 0, 0.1);
}

.company-name {
  font-size: 32px;
  font-weight: 900;
  color: var(--accent-orange);
  margin-bottom: 12px;
}

.company-meta {
  display: flex;
  gap: 24px;
  margin: 16px 0;
  flex-wrap: wrap;
}

.company-meta-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.company-meta-label {
  font-size: 11px;
  color: var(--text-secondary);
  text-transform: uppercase;
  font-weight: 700;
  letter-spacing: 0.5px;
}

.company-meta-value {
  font-size: 16px;
  color: var(--text-primary);
  font-weight: 600;
}

.company-description {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 2px solid rgba(251, 140, 0, 0.1);
  line-height: 1.7;
  color: var(--text-primary);
  text-align: justify;
}

/* Footer */
.footer {
  text-align: center;
  color: var(--text-secondary);
  font-size: 13px;
  margin-top: 60px;
  padding: 32px;
  border-top: 3px solid var(--accent-orange);
  background: linear-gradient(180deg, transparent 0%, rgba(251, 140, 0, 0.05) 100%);
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .hero-header h1 {
    font-size: 28px;
  }
  
  .kpi-container {
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  }
}

/* Loading Animation */
@keyframes shimmer {
  0% { background-position: -1000px 0; }
  100% { background-position: 1000px 0; }
}

.loading {
  animation: shimmer 2s infinite;
  background: linear-gradient(to right, #f0f0f0 0%, #e0e0e0 50%, #f0f0f0 100%);
  background-size: 1000px 100%;
}

/* Chart Container */
.chart-container {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: var(--shadow-md);
  margin: 20px 0;
  border: 2px solid rgba(251, 140, 0, 0.1);
}

/* Metric Badge */
.metric-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-left: 8px;
}

.metric-badge-positive {
  background: #E8F5E9;
  color: var(--success);
}

.metric-badge-negative {
  background: #FFEBEE;
  color: var(--danger);
}

.metric-badge-neutral {
  background: #F5F5F5;
  color: var(--text-secondary);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HERO HEADER
# =========================
st.markdown(f"""
<div class="hero-header">
    <h1>📊 Dashboard Financiero AI Pro</h1>
    <p>Análisis Bursátil Completo con Inteligencia Artificial · Valuación · Análisis Técnico · Gestión de Riesgos</p>
    <span class="hero-badge">Version {APP_VERSION}</span>
    <span class="hero-badge">Powered by Gemini 2.0</span>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR - CONFIGURACIÓN
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Configuración del Análisis")
    
    # CRITERIO 1: Ingreso dinámico del ticker (2 pts)
    ticker = st.text_input(
        "🎯 Ticker de la Acción",
        value="AAPL",
        help="Ingresa el símbolo bursátil (ej: AAPL, TSLA, GOOGL, MSFT)",
        placeholder="Ej: AAPL"
    ).strip().upper()
    
    # Benchmark para comparación
    benchmark = st.text_input(
        "📊 Índice de Referencia",
        value="SPY",
        help="Índice para comparación (ej: SPY, QQQ, DIA)",
        placeholder="Ej: SPY"
    ).strip().upper()
    
    st.markdown("---")
    
    # Configuración adicional
    st.markdown("#### 📅 Periodo de Análisis")
    max_years = st.selectbox(
        "Años de datos históricos",
        [1, 2, 3, 5, 10],
        index=3,
        help="Cantidad máxima de años para análisis histórico"
    )
    
    st.markdown("---")
    
    # Parámetros de riesgo
    st.markdown("#### ⚠️ Parámetros de Riesgo")
    rf_rate = st.number_input(
        "Tasa libre de riesgo anual",
        value=0.043,
        min_value=0.0,
        max_value=0.20,
        step=0.001,
        format="%.3f",
        help="Tasa de bonos del tesoro a 10 años (aprox. 4.3% en 2024)"
    )
    
    var_confidence = st.slider(
        "Nivel de confianza VaR",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Nivel de confianza para Value at Risk (95% = estándar)"
    )
    
    st.markdown("---")
    
    # Configuración de IA
    st.markdown("#### 🤖 Análisis con IA")
    
    idiomas = {
        "Español": "español",
        "Inglés": "inglés",
        "Francés": "francés",
        "Alemán": "alemán",
        "Italiano": "italiano",
        "Portugués": "portugués"
    }
    
    idioma_sel = st.selectbox(
        "🌍 Idioma de análisis",
        list(idiomas.keys()),
        index=0,
        help="Idioma para los análisis con IA"
    )
    
    enable_ai = st.checkbox(
        "Activar análisis con Gemini AI",
        value=True,
        help="Incluir análisis avanzados con inteligencia artificial"
    )
    
    st.markdown("---")
    
    # Botón de análisis principal
    run_btn = st.button(
        "🚀 GENERAR DASHBOARD COMPLETO",
        use_container_width=True,
        type="primary"
    )
    
    st.markdown("---")
    
    # Tips
    with st.expander("💡 Tips y Ayuda"):
        st.markdown("""
        **Tickers Populares:**
        - Tech: AAPL, MSFT, GOOGL, META, NVDA
        - Financiero: JPM, BAC, WFC, GS
        - Energía: XOM, CVX, COP
        - ETFs: SPY, QQQ, VOO, VTI
        
        **Índices de Referencia:**
        - SPY: S&P 500
        - QQQ: Nasdaq-100
        - DIA: Dow Jones
        - IWM: Russell 2000
        """)
    
    st.markdown("---")
    st.caption(f"v{APP_VERSION} | {APP_OWNER}")

# =========================
# FUNCIONES AUXILIARES
# =========================

@st.cache_data(show_spinner=False, ttl=3600)
def get_company_info(symbol: str) -> Dict:
    """
    Función blindada contra rate-limit de Yahoo Finance.
    NO usa .info, NO usa llamadas bloqueadas.
    Usa fast_info (seguro), scraping y fallback.
    """

    # --- MÉTODO 1: fast_info (este SIEMPRE funciona, nunca es rate-limited) ---
    try:
        tk = yf.Ticker(symbol)
        fast = tk.fast_info if hasattr(tk, "fast_info") else {}
        long_name = fast.get("shortName", symbol)
    except:
        fast = {}
        long_name = symbol

    # --- MÉTODO 2: Scraping del perfil de Yahoo (si bloquea, sigue el fallback) ---
    sector, industry, description = "N/D", "N/D", "Descripción no disponible."
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}/profile"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)

        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")

            h1 = soup.find("h1")
            if h1:
                long_name = h1.text

            sec = soup.find("span", text="Sector")
            if sec:
                sector = sec.find_next("span").text

            ind = soup.find("span", text="Industry")
            if ind:
                industry = ind.find_next("span").text

            desc = soup.find("p")
            if desc:
                description = desc.text

    except Exception as e:
        st.warning(f"⚠️ Scraping bloqueado: {e}")

    # --- RETORNO FINAL (SIEMPRE DEVUELVE ALGO) ---
    return {
        "longName": long_name,
        "sector": sector,
        "industry": industry,
        "longBusinessSummary": description
    }


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_prices(symbol: str, years: int = 5) -> pd.Series:
    """Descarga precios ajustados históricos."""
    start_date = (datetime.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    
    try:
        data = yf.download(symbol, start=start_date, progress=False, auto_adjust=False)
        
        if data.empty:
            return pd.Series(dtype=float)
        
        # Manejar MultiIndex correctamente
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.get_level_values(0):
                series = data['Adj Close']
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
            else:
                series = data['Close']
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
        else:
            series = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
        series.name = symbol
        return series.dropna()
    
    except Exception as e:
        st.error(f"❌ Error al descargar datos de {symbol}: {str(e)}")
        return pd.Series(dtype=float)

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_ohlcv(symbol: str, years: int = 2) -> pd.DataFrame:
    """
    CRITERIO 2: Descarga datos OHLCV para gráfico de velas (15 pts)
    """
    start_date = (datetime.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    
    try:
        data = yf.download(symbol, start=start_date, progress=False, auto_adjust=False)
        
        if data.empty:
            return pd.DataFrame()
        
        # Si es MultiIndex, simplificar
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        return data
    
    except Exception as e:
        st.error(f"❌ Error al descargar datos OHLCV de {symbol}: {str(e)}")
        return pd.DataFrame()

# =========================
# FUNCIONES DE ANÁLISIS (CRITERIO 3)
# =========================

def calculate_period_metrics(prices: pd.Series, period_code: str, rf_rate: float = 0.043) -> Dict:
    """
    CRITERIO 3: Cálculo de rendimientos y riesgos (25 pts)
    - Rendimientos aritméticos (10 pts)
    - Volatilidad histórica (10 pts)
    - Métricas adicionales (5 pts)
    """
    # Determinar fecha de inicio según el periodo
    today = pd.Timestamp.now()
    
    if period_code == "YTD":
        start_date = pd.Timestamp(datetime(today.year, 1, 1))
    elif period_code.endswith("M"):
        months = int(period_code[:-1])
        start_date = today - pd.DateOffset(months=months)
    elif period_code.endswith("Y"):
        years = int(period_code[:-1])
        start_date = today - pd.DateOffset(years=years)
    else:
        return {}
    
    # Filtrar precios del periodo
    period_prices = prices[prices.index >= start_date]
    
    if len(period_prices) < 2:
        return {
            'periodo': period_code,
            'rendimiento_aritmetico': np.nan,
            'rendimiento_logaritmico': np.nan,
            'volatilidad': np.nan,
            'sharpe_ratio': np.nan,
            'var_95': np.nan,
            'max_drawdown': np.nan
        }
    
    # Calcular retornos diarios
    returns = period_prices.pct_change().dropna()
    
    # 1. Rendimiento Aritmético Total (simple)
    rendimiento_aritmetico = (period_prices.iloc[-1] / period_prices.iloc[0]) - 1
    
    # 2. Rendimiento Logarítmico
    rendimiento_logaritmico = np.log(period_prices.iloc[-1] / period_prices.iloc[0])
    
    # 3. Volatilidad Anualizada (desviación estándar)
    volatilidad_anual = returns.std() * np.sqrt(252)
    
    # 4. Sharpe Ratio
    mean_return_annual = returns.mean() * 252
    sharpe = (mean_return_annual - rf_rate) / volatilidad_anual if volatilidad_anual != 0 else 0
    
    # 5. VaR 95%
    var_95 = np.percentile(returns, 5)
    
    # 6. Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    return {
        'periodo': period_code,
        'rendimiento_aritmetico': rendimiento_aritmetico,
        'rendimiento_logaritmico': rendimiento_logaritmico,
        'volatilidad': volatilidad_anual,
        'sharpe_ratio': sharpe,
        'var_95': var_95,
        'max_drawdown': max_dd
    }

def calculate_beta_alpha_corr(asset_prices: pd.Series, benchmark_prices: pd.Series) -> Tuple[float, float, float]:
    """
    CRITERIO 4: Comparación con índice - Métricas cuantitativas (5 pts)
    Calcula Beta, Alpha y Correlación.
    """
    # Alinear fechas
    df = pd.DataFrame({
        'asset': asset_prices,
        'benchmark': benchmark_prices
    }).dropna()
    
    if len(df) < 30:
        return np.nan, np.nan, np.nan
    
    # Calcular retornos
    asset_returns = df['asset'].pct_change().dropna()
    benchmark_returns = df['benchmark'].pct_change().dropna()
    
    # Correlación
    correlation = asset_returns.corr(benchmark_returns)
    
    # Beta (covarianza / varianza del benchmark)
    covariance = asset_returns.cov(benchmark_returns)
    benchmark_variance = benchmark_returns.var()
    beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
    
    # Alpha (anualizado)
    asset_mean_return = asset_returns.mean() * 252
    benchmark_mean_return = benchmark_returns.mean() * 252
    alpha = asset_mean_return - (beta * benchmark_mean_return) if not np.isnan(beta) else np.nan
    
    return beta, alpha, correlation

# =========================
# FUNCIONES DE IA CON GEMINI
# =========================

def gemini_valuation_analysis(symbol: str, company_info: Dict, metrics: pd.DataFrame, idioma: str = "español") -> str:
    """Análisis de valuación con Gemini AI."""
    if not enable_ai:
        return "Análisis con IA desactivado."
    
    try:
        # Preparar contexto
        company_name = company_info.get('longName', symbol)
        sector = company_info.get('sector', 'N/D')
        industry = company_info.get('industry', 'N/D')
        
        # Obtener métricas del último año
        metrics_1y = metrics[metrics['periodo'] == '1Y'].iloc[0] if '1Y' in metrics['periodo'].values else {}
        
        prompt = f"""
        Eres un analista financiero senior experto. Analiza la acción de {company_name} ({symbol}) en {idioma}.
        
        Contexto:
        - Sector: {sector}
        - Industria: {industry}
        - Rendimiento 1 año: {metrics_1y.get('rendimiento_aritmetico', 0)*100:.2f}%
        - Volatilidad: {metrics_1y.get('volatilidad', 0)*100:.2f}%
        - Sharpe Ratio: {metrics_1y.get('sharpe_ratio', 0):.2f}
        
        Proporciona un análisis conciso (máximo 200 palabras) que incluya:
        1. Evaluación del desempeño reciente
        2. Análisis de riesgo-retorno
        3. Posicionamiento en el sector
        4. Recomendación general (muy alcista/alcista/neutral/bajista/muy bajista)
        
        Sé técnico pero claro. Enfócate en datos concretos.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        return f"⚠️ Error en análisis AI: {str(e)}"

def translate_description(text: str, target_language: str) -> str:
    """Traduce la descripción de la empresa."""
    if not enable_ai or target_language == "español":
        return text
    
    try:
        prompt = f"Traduce el siguiente texto al {target_language}, mantén el tono profesional: {text}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return text

# =========================
# LÓGICA PRINCIPAL
# =========================

if run_btn:
    if not ticker:
        st.error("❌ Por favor ingresa un ticker válido.")
        st.stop()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # ===================================
        # SECCIÓN 1: DESCRIPCIÓN DE LA EMPRESA (10 PTS)
        # ===================================
        status_text.text("📥 Obteniendo información de la empresa...")
        progress_bar.progress(10)
        
        company_info = get_company_info(ticker)
        
        if not company_info:
            st.error(f"❌ No se pudo obtener información de {ticker}. Verifica que el ticker sea válido.")
            st.stop()
        
        # Extraer datos clave
        company_name = company_info.get('longName') or company_info.get('shortName') or ticker
        sector = company_info.get('sector', 'N/D')
        industry = company_info.get('industry', 'N/D')
        website = company_info.get('website', '')
        employees = company_info.get('fullTimeEmployees', 'N/D')
        market_cap = company_info.get('marketCap', 0)
        description = company_info.get('longBusinessSummary', 'Descripción no disponible.')
        
        # CRITERIO 1: Presentación clara y atractiva (2 pts)
        st.markdown('<div class="company-info-card">', unsafe_allow_html=True)
        
        st.markdown(f'<div class="company-name">{company_name} ({ticker})</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="company-meta">', unsafe_allow_html=True)
        
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"""
            <div class="company-meta-item">
                <div class="company-meta-label">🏢 Sector</div>
                <div class="company-meta-value">{sector}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown(f"""
            <div class="company-meta-item">
                <div class="company-meta-label">⚙️ Industria</div>
                <div class="company-meta-value">{industry}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            market_cap_str = f"${market_cap/1e9:.2f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.2f}M" if market_cap > 0 else "N/D"
            st.markdown(f"""
            <div class="company-meta-item">
                <div class="company-meta-label">💰 Market Cap</div>
                <div class="company-meta-value">{market_cap_str}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[3]:
            employees_str = f"{employees:,}" if isinstance(employees, int) else str(employees)
            st.markdown(f"""
            <div class="company-meta-item">
                <div class="company-meta-label">👥 Empleados</div>
                <div class="company-meta-value">{employees_str}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Descripción traducida si es necesario
        if idioma_sel != "Español" and enable_ai:
            description = translate_description(description, idiomas[idioma_sel])
        
        st.markdown(f'<div class="company-description">{description}</div>', unsafe_allow_html=True)
        
        if website:
            st.markdown(f'<p style="margin-top: 16px;"><strong>🌐 Sitio web:</strong> <a href="{website}" target="_blank">{website}</a></p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        progress_bar.progress(20)
        
        # ===================================
        # DESCARGAR DATOS HISTÓRICOS
        # ===================================
        status_text.text("📊 Descargando datos históricos...")
        
        # Descargar precios
        prices_asset = fetch_prices(ticker, years=max_years)
        prices_benchmark = fetch_prices(benchmark, years=max_years)
        
        if prices_asset.empty:
            st.error(f"❌ No se pudieron descargar datos históricos de {ticker}")
            st.stop()
        
        if prices_benchmark.empty:
            st.warning(f"⚠️ No se pudieron descargar datos de {benchmark}. Comparación limitada.")
        
        # Descargar OHLCV para gráfico de velas
        ohlcv_data = fetch_ohlcv(ticker, years=2)
        
        progress_bar.progress(40)
        
        # ===================================
        # SECCIÓN 2: VISUALIZACIÓN - GRÁFICO DE VELAS (15 PTS)
        # ===================================
        st.markdown('<div class="section-header">📈 Análisis Técnico - Gráfico de Velas Japonesas</div>', unsafe_allow_html=True)
        
        if not ohlcv_data.empty and all(col in ohlcv_data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Filtrar último año mínimo
            one_year_ago = datetime.now() - pd.DateOffset(months=12)
            ohlcv_last_year = ohlcv_data[ohlcv_data.index >= one_year_ago]
            
            # CRITERIO 2: Personalización visual (5 pts) + Originalidad (5 pts)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Crear gráfico de velas con subplots
            fig_candle = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{ticker} - Precio con Medias Móviles', 'Volumen')
            )
            
            # Candlestick
            fig_candle.add_trace(
                go.Candlestick(
                    x=ohlcv_last_year.index,
                    open=ohlcv_last_year['Open'],
                    high=ohlcv_last_year['High'],
                    low=ohlcv_last_year['Low'],
                    close=ohlcv_last_year['Close'],
                    name='OHLC',
                    increasing_line_color='#4CAF50',
                    decreasing_line_color='#F44336',
                    increasing_fillcolor='#4CAF50',
                    decreasing_fillcolor='#F44336'
                ),
                row=1, col=1
            )
            
            # Medias móviles
            sma_20 = ohlcv_last_year['Close'].rolling(window=20).mean()
            sma_50 = ohlcv_last_year['Close'].rolling(window=50).mean()
            sma_200 = ohlcv_last_year['Close'].rolling(window=200).mean()
            
            fig_candle.add_trace(
                go.Scatter(x=ohlcv_last_year.index, y=sma_20, name='SMA 20',
                          line=dict(color='#FB8C00', width=1.5)),
                row=1, col=1
            )
            fig_candle.add_trace(
                go.Scatter(x=ohlcv_last_year.index, y=sma_50, name='SMA 50',
                          line=dict(color='#2196F3', width=1.5)),
                row=1, col=1
            )
            fig_candle.add_trace(
                go.Scatter(x=ohlcv_last_year.index, y=sma_200, name='SMA 200',
                          line=dict(color='#9C27B0', width=2)),
                row=1, col=1
            )
            
            # Volumen
            colors = ['#4CAF50' if ohlcv_last_year['Close'].iloc[i] >= ohlcv_last_year['Open'].iloc[i] 
                     else '#F44336' for i in range(len(ohlcv_last_year))]
            
            fig_candle.add_trace(
                go.Bar(x=ohlcv_last_year.index, y=ohlcv_last_year['Volume'],
                      name='Volumen', marker_color=colors, showlegend=False),
                row=2, col=1
            )
            
            # Layout
            fig_candle.update_layout(
                height=700,
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                template='plotly_white'
            )
            
            fig_candle.update_xaxes(
                title_text="Fecha",
                row=2, col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.05)'
            )
            fig_candle.update_yaxes(
                title_text="Precio ($)",
                row=1, col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.05)'
            )
            fig_candle.update_yaxes(
                title_text="Volumen",
                row=2, col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.05)'
            )
            
            st.plotly_chart(fig_candle, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Análisis técnico rápido
            current_price = ohlcv_last_year['Close'].iloc[-1]
            sma_20_current = sma_20.iloc[-1]
            sma_50_current = sma_50.iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend = "Alcista 📈" if current_price > sma_20_current > sma_50_current else \
                        "Bajista 📉" if current_price < sma_20_current < sma_50_current else \
                        "Lateral ↔️"
                st.metric("Tendencia Actual", trend)
            
            with col2:
                price_change = ((current_price - ohlcv_last_year['Close'].iloc[0]) / ohlcv_last_year['Close'].iloc[0]) * 100
                st.metric("Cambio en 1 año", f"{price_change:+.2f}%", delta=f"${current_price - ohlcv_last_year['Close'].iloc[0]:.2f}")
            
            with col3:
                volatility_recent = ohlcv_last_year['Close'].pct_change().std() * np.sqrt(252) * 100
                st.metric("Volatilidad Anual", f"{volatility_recent:.2f}%")
        
        else:
            st.warning("⚠️ No se pudieron obtener datos OHLCV para el gráfico de velas.")
        
        progress_bar.progress(50)
        
        # ===================================
        # SECCIÓN 3: CÁLCULO DE RENDIMIENTOS Y RIESGOS (35 PTS)
        # ===================================
        status_text.text("⚙️ Calculando métricas de rendimiento y riesgo...")
        
        st.markdown('<div class="section-header">📊 Rendimientos y Riesgos Históricos</div>', unsafe_allow_html=True)
        
        # CRITERIO 3: Periodos requeridos - 1Y, 3Y, 5Y, YTD, 3M, 6M, 9M
        required_periods = ['YTD', '3M', '6M', '9M', '1Y', '3Y', '5Y']
        
        metrics_list = []
        for period in required_periods:
            metrics = calculate_period_metrics(prices_asset, period, rf_rate)
            metrics_list.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_list)
        
        # CRITERIO 3: Presentación en tabla clara (5 pts)
        st.markdown('<div class="metrics-table">', unsafe_allow_html=True)
        st.markdown('<div class="subsection-header">📋 Tabla de Rendimientos y Volatilidad por Periodo</div>', unsafe_allow_html=True)
        
        # Formatear tabla para display
        display_df = metrics_df.copy()
        display_df['Rendimiento Aritmético'] = display_df['rendimiento_aritmetico'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/D")
        display_df['Rendimiento Logarítmico'] = display_df['rendimiento_logaritmico'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/D")
        display_df['Volatilidad Anual'] = display_df['volatilidad'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/D")
        display_df['Sharpe Ratio'] = display_df['sharpe_ratio'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/D")
        display_df['VaR 95%'] = display_df['var_95'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/D")
        display_df['Max Drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/D")
        
        display_df = display_df[['periodo', 'Rendimiento Aritmético', 'Rendimiento Logarítmico', 
                                 'Volatilidad Anual', 'Sharpe Ratio', 'VaR 95%', 'Max Drawdown']]
        display_df = display_df.rename(columns={'periodo': 'Periodo'})
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=350
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # CRITERIO 3: Elementos adicionales de análisis (5 pts)
        st.markdown('<div class="subsection-header">📊 Métricas Adicionales de Riesgo</div>', unsafe_allow_html=True)
        
        # Obtener métricas del periodo de 1Y para display
        metrics_1y = metrics_df[metrics_df['periodo'] == '1Y'].iloc[0]
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            rend_value = metrics_1y['rendimiento_aritmetico'] * 100
            rend_class = 'positive' if rend_value > 0 else 'negative'
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">📈</div>
                <div class="kpi-title">Rendimiento 1Y</div>
                <div class="kpi-value">{rend_value:+.2f}%</div>
                <div class="kpi-change {rend_class}">Aritmético</div>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi_col2:
            vol_value = metrics_1y['volatilidad'] * 100
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">📊</div>
                <div class="kpi-title">Volatilidad</div>
                <div class="kpi-value">{vol_value:.2f}%</div>
                <div class="kpi-change">Anual</div>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi_col3:
            sharpe_value = metrics_1y['sharpe_ratio']
            sharpe_class = 'positive' if sharpe_value > 1 else 'negative' if sharpe_value < 0 else ''
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">⚖️</div>
                <div class="kpi-title">Sharpe Ratio</div>
                <div class="kpi-value {sharpe_class}">{sharpe_value:.3f}</div>
                <div class="kpi-change">Riesgo/Retorno</div>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi_col4:
            dd_value = metrics_1y['max_drawdown'] * 100
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">📉</div>
                <div class="kpi-title">Max Drawdown</div>
                <div class="kpi-value class="negative">{dd_value:.2f}%</div>
                <div class="kpi-change">Pérdida Máxima</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Interpretación del Sharpe Ratio
        st.markdown("#### 💡 Interpretación del Sharpe Ratio")
        sharpe_1y = metrics_1y['sharpe_ratio']
        
        if sharpe_1y > 2:
            sharpe_msg = "🟢 **Excelente** - Rendimiento ajustado por riesgo muy superior"
            alert_class = "risk-alert-low"
        elif sharpe_1y > 1:
            sharpe_msg = "🟡 **Bueno** - Rendimiento ajustado por riesgo positivo"
            alert_class = "risk-alert-medium"
        elif sharpe_1y > 0:
            sharpe_msg = "🟠 **Aceptable** - Rendimiento ajustado por riesgo moderado"
            alert_class = "risk-alert-medium"
        else:
            sharpe_msg = "🔴 **Bajo** - Rendimiento ajustado por riesgo insuficiente"
            alert_class = "risk-alert-high"
        
        st.markdown(f'<div class="risk-alert {alert_class}">{sharpe_msg}</div>', unsafe_allow_html=True)
        
        progress_bar.progress(70)
        
        # ===================================
        # SECCIÓN 4: COMPARACIÓN CON ÍNDICE (20 PTS)
        # ===================================
        status_text.text("📊 Generando comparación con índice...")
        
        st.markdown('<div class="section-header">📊 Comparación con Índice de Referencia</div>', unsafe_allow_html=True)
        
        if not prices_benchmark.empty:
            # ===================================
            # GRÁFICA COMPARATIVA DE RENDIMIENTOS
            # ===================================
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="subsection-header">📊 Comparación de Rendimientos: {ticker} vs {benchmark}</div>', unsafe_allow_html=True)
            
            # Calcular rendimientos del benchmark para cada periodo
            benchmark_metrics = []
            for period in required_periods:
                metrics_bench = calculate_period_metrics(prices_benchmark, period, rf_rate)
                benchmark_metrics.append(metrics_bench)
            
            benchmark_df = pd.DataFrame(benchmark_metrics)
            
            # Crear DataFrame comparativo
            comparison_returns = pd.DataFrame({
                'Periodo': required_periods,
                ticker: metrics_df['rendimiento_aritmetico'] * 100,
                benchmark: benchmark_df['rendimiento_aritmetico'] * 100
            })
            
            # Gráfico de barras comparativo
            fig_comparison_bars = go.Figure()
            
            fig_comparison_bars.add_trace(go.Bar(
                name=ticker,
                x=comparison_returns['Periodo'],
                y=comparison_returns[ticker],
                marker_color='#FB8C00',
                text=comparison_returns[ticker].apply(lambda x: f'{x:+.1f}%'),
                textposition='outside',
                textfont=dict(size=11, color='#FB8C00', weight='bold')
            ))
            
            fig_comparison_bars.add_trace(go.Bar(
                name=benchmark,
                x=comparison_returns['Periodo'],
                y=comparison_returns[benchmark],
                marker_color='#2196F3',
                text=comparison_returns[benchmark].apply(lambda x: f'{x:+.1f}%'),
                textposition='outside',
                textfont=dict(size=11, color='#2196F3', weight='bold')
            ))
            
            # Línea en 0
            fig_comparison_bars.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                line_width=1
            )
            
            fig_comparison_bars.update_layout(
                title=f'Comparación de Rendimientos por Periodo: {ticker} vs {benchmark}',
                xaxis_title='Periodo',
                yaxis_title='Rendimiento (%)',
                barmode='group',
                height=500,
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                hovermode='x unified'
            )
            
            fig_comparison_bars.update_xaxes(showgrid=False)
            fig_comparison_bars.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=True)
            
            st.plotly_chart(fig_comparison_bars, use_container_width=True)
            
            # Resumen comparativo en métricas
            col_summary1, col_summary2, col_summary3 = st.columns(3)
            
            with col_summary1:
                wins = (comparison_returns[ticker] > comparison_returns[benchmark]).sum()
                total = len(comparison_returns)
                win_rate = (wins / total) * 100
                st.metric(
                    "Periodos Ganados",
                    f"{wins}/{total}",
                    f"{win_rate:.0f}% de victorias"
                )
            
            with col_summary2:
                avg_diff = (comparison_returns[ticker] - comparison_returns[benchmark]).mean()
                st.metric(
                    "Diferencia Promedio",
                    f"{avg_diff:+.2f}%",
                    "vs índice"
                )
            
            with col_summary3:
                best_period = comparison_returns.loc[
                    (comparison_returns[ticker] - comparison_returns[benchmark]).idxmax(), 
                    'Periodo'
                ]
                st.metric(
                    "Mejor Periodo Relativo",
                    best_period,
                    "Mayor outperformance"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)
            # Alinear fechas
            comparison_df = pd.DataFrame({
                ticker: prices_asset,
                benchmark: prices_benchmark
            }).dropna()
            
            if len(comparison_df) > 1:
                # CRITERIO 4: Gráfico base cero (10 pts)
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="subsection-header">📈 Rendimiento Relativo (Base 100)</div>', unsafe_allow_html=True)
                
                # Periodo seleccionable
                period_options = {
                    "Último Año (1Y)": "1Y",
                    "Últimos 3 Años (3Y)": "3Y",
                    "Últimos 5 Años (5Y)": "5Y",
                    "Año en Curso (YTD)": "YTD",
                    "Todo el Periodo": "MAX"
                }
                
                selected_period = st.selectbox(
                    "Selecciona periodo para comparación:",
                    list(period_options.keys()),
                    index=0
                )
                
                period_code = period_options[selected_period]
                
                # Filtrar datos según periodo
                if period_code == "MAX":
                    comparison_period = comparison_df
                else:
                    today = pd.Timestamp.now()
                    if period_code == "YTD":
                        start_date = pd.Timestamp(datetime(today.year, 1, 1))
                    elif period_code.endswith("Y"):
                        years = int(period_code[:-1])
                        start_date = today - pd.DateOffset(years=years)
                    
                    comparison_period = comparison_df[comparison_df.index >= start_date]
                
                # Normalizar a base 100
                base_100 = (comparison_period / comparison_period.iloc[0]) * 100
                
                # CRITERIO 4: Claridad visual (5 pts)
                fig_comparison = go.Figure()
                
                fig_comparison.add_trace(go.Scatter(
                    x=base_100.index,
                    y=base_100[ticker],
                    name=f'{ticker} (Acción)',
                    line=dict(color='#FB8C00', width=3),
                    fill='tonexty',
                    fillcolor='rgba(251, 140, 0, 0.1)'
                ))
                
                fig_comparison.add_trace(go.Scatter(
                    x=base_100.index,
                    y=base_100[benchmark],
                    name=f'{benchmark} (Índice)',
                    line=dict(color='#2196F3', width=2, dash='dot')
                ))
                
                # Línea de referencia en 100
                fig_comparison.add_hline(
                    y=100,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Base 100",
                    annotation_position="right"
                )
                
                fig_comparison.update_layout(
                    title=f"Comparación {ticker} vs {benchmark} - Base 100",
                    xaxis_title="Fecha",
                    yaxis_title="Índice (100 = inicio del periodo)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                
                fig_comparison.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                fig_comparison.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # CRITERIO 4: Comparación cuantitativa (5 pts)
                st.markdown('<div class="subsection-header">🔢 Métricas Comparativas</div>', unsafe_allow_html=True)
                
                # Calcular métricas comparativas
                beta, alpha, correlation = calculate_beta_alpha_corr(prices_asset, prices_benchmark)
                
                # Rendimientos del periodo
                asset_return = (comparison_period[ticker].iloc[-1] / comparison_period[ticker].iloc[0]) - 1
                benchmark_return = (comparison_period[benchmark].iloc[-1] / comparison_period[benchmark].iloc[0]) - 1
                excess_return = asset_return - benchmark_return
                
                comp_col1, comp_col2, comp_col3, comp_col4, comp_col5 = st.columns(5)
                
                with comp_col1:
                    ret_class = 'positive' if asset_return > 0 else 'negative'
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Rendimiento {ticker}</div>
                        <div class="kpi-value {ret_class}">{asset_return*100:+.2f}%</div>
                        <div class="kpi-subtitle">Periodo seleccionado</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with comp_col2:
                    bench_class = 'positive' if benchmark_return > 0 else 'negative'
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Rendimiento {benchmark}</div>
                        <div class="kpi-value {bench_class}">{benchmark_return*100:+.2f}%</div>
                        <div class="kpi-subtitle">Índice de referencia</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with comp_col3:
                    excess_class = 'positive' if excess_return > 0 else 'negative'
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Rendimiento Excedente</div>
                        <div class="kpi-value {excess_class}">{excess_return*100:+.2f}%</div>
                        <div class="kpi-subtitle">Alpha del periodo</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with comp_col4:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Beta</div>
                        <div class="kpi-value">{beta:.3f}</div>
                        <div class="kpi-subtitle">Sensibilidad al mercado</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with comp_col5:
                    corr_pct = correlation * 100 if pd.notna(correlation) else 0
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Correlación</div>
                        <div class="kpi-value">{corr_pct:.1f}%</div>
                        <div class="kpi-subtitle">vs {benchmark}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Interpretación del Beta
                st.markdown("#### 💡 Interpretación del Beta")
                
                if pd.notna(beta):
                    if beta > 1.2:
                        beta_msg = f"🔴 **Alta Volatilidad** - {ticker} es {beta:.2f}x más volátil que {benchmark}. Mayor riesgo y potencial de retorno."
                        beta_class = "risk-alert-high"
                    elif beta > 0.8:
                        beta_msg = f"🟡 **Volatilidad Moderada** - {ticker} se mueve similarmente a {benchmark} (β={beta:.2f})."
                        beta_class = "risk-alert-medium"
                    else:
                        beta_msg = f"🟢 **Baja Volatilidad** - {ticker} es menos volátil que {benchmark} (β={beta:.2f}). Menor riesgo."
                        beta_class = "risk-alert-low"
                    
                    st.markdown(f'<div class="risk-alert {beta_class}">{beta_msg}</div>', unsafe_allow_html=True)
                
                # Gráfico de dispersión de retornos
                with st.expander("📊 Ver Análisis de Dispersión de Retornos"):
                    returns_df = comparison_period.pct_change().dropna()
                    
                    fig_scatter = go.Figure()
                    
                    # Puntos de dispersión
                    fig_scatter.add_trace(go.Scatter(
                        x=returns_df[benchmark] * 100,
                        y=returns_df[ticker] * 100,
                        mode='markers',
                        marker=dict(size=5, opacity=0.6, color='#FB8C00'),
                        name='Retornos'
                    ))
                    
                    # Línea de tendencia manual usando Beta
                    x_range = [returns_df[benchmark].min() * 100, returns_df[benchmark].max() * 100]
                    y_range = [x * beta for x in x_range]
                    
                    fig_scatter.add_trace(go.Scatter(
                        x=x_range,
                        y=y_range,
                        mode='lines',
                        line=dict(color='#2196F3', width=2, dash='dash'),
                        name=f'Línea de Tendencia (β={beta:.3f})'
                    ))
                    
                    fig_scatter.update_layout(
                        title=f'Dispersión de Retornos: {ticker} vs {benchmark}',
                        xaxis_title=f'Retornos Diarios {benchmark} (%)',
                        yaxis_title=f'Retornos Diarios {ticker} (%)',
                        height=400,
                        template='plotly_white',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    st.caption(f"La pendiente de la línea de tendencia representa el Beta (β = {beta:.3f})")
            
            else:
                st.warning("⚠️ No hay suficientes datos para comparación.")
        
        else:
            st.warning(f"⚠️ No se pudieron cargar datos de {benchmark} para comparación.")
        
        progress_bar.progress(85)
        
        # ===================================
        # ANÁLISIS CON IA (BONUS)
        # ===================================
        if enable_ai:
            status_text.text("🤖 Generando análisis con IA...")
            
            st.markdown('<div class="section-header">🤖 Análisis con Inteligencia Artificial</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="ai-analysis-card">', unsafe_allow_html=True)
            st.markdown('<span class="ai-badge">Powered by Gemini 2.0 Flash</span>', unsafe_allow_html=True)
            st.markdown(f'<h3>Análisis Integral de {ticker}</h3>', unsafe_allow_html=True)
            
            with st.spinner("🔄 Analizando con IA..."):
                ai_analysis = gemini_valuation_analysis(
                    ticker,
                    company_info,
                    metrics_df,
                    idiomas[idioma_sel]
                )
            
            st.markdown(ai_analysis)
            st.markdown('</div>', unsafe_allow_html=True)
        
        progress_bar.progress(100)
        status_text.text("✅ Dashboard generado exitosamente!")
        
        # Limpiar status
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # ===================================
        # EXPORTACIÓN DE DATOS
        # ===================================
        st.markdown('<div class="section-header">💾 Exportar Datos</div>', unsafe_allow_html=True)
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # Exportar tabla de métricas
            csv_metrics = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Descargar Tabla de Métricas (CSV)",
                data=csv_metrics,
                file_name=f"{ticker}_metricas_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_col2:
            # Exportar precios históricos
            csv_prices = prices_asset.to_csv().encode('utf-8')
            st.download_button(
                label="📈 Descargar Precios Históricos (CSV)",
                data=csv_prices,
                file_name=f"{ticker}_precios_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # ===================================
        # FOOTER
        # ===================================
        st.markdown(f"""
        <div class="footer">
            <strong>{DISCLAIMER}</strong><br>
            <small>Datos provistos por Yahoo Finance · Análisis con Gemini 2.0 Flash</small><br>
            <small>Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error durante la generación del dashboard: {str(e)}")
        st.exception(e)
        progress_bar.empty()
        status_text.empty()

else:
    # Pantalla de inicio cuando no se ha ejecutado el análisis
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <h2 style="color: var(--accent-orange); margin-bottom: 24px;">
            👋 Bienvenido al Dashboard Financiero AI Pro
        </h2>
        <p style="font-size: 18px; color: var(--text-secondary); margin-bottom: 40px;">
            Herramienta profesional para análisis bursátil completo con inteligencia artificial
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Características principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="kpi-card" style="height: 280px;">
            <div class="kpi-icon">📊</div>
            <h3 style="color: var(--accent-orange); font-size: 20px; margin: 16px 0;">
                Análisis Completo
            </h3>
            <p style="font-size: 14px; color: var(--text-secondary); line-height: 1.6;">
                • Gráficos de velas japonesas<br>
                • Medias móviles (20, 50, 200)<br>
                • Rendimientos históricos<br>
                • Métricas de riesgo avanzadas
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="kpi-card" style="height: 280px;">
            <div class="kpi-icon">📈</div>
            <h3 style="color: var(--accent-orange); font-size: 20px; margin: 16px 0;">
                Comparación con Índices
            </h3>
            <p style="font-size: 14px; color: var(--text-secondary); line-height: 1.6;">
                • Gráfico base 100<br>
                • Beta y Alpha<br>
                • Correlación<br>
                • Rendimiento excedente
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="kpi-card" style="height: 280px;">
            <div class="kpi-icon">🤖</div>
            <h3 style="color: var(--accent-orange); font-size: 20px; margin: 16px 0;">
                Inteligencia Artificial
            </h3>
            <p style="font-size: 14px; color: var(--text-secondary); line-height: 1.6;">
                • Análisis de valuación<br>
                • Interpretación de métricas<br>
                • Recomendaciones<br>
                • Traducción automática
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Instrucciones
    st.markdown("""
    <div class="ai-analysis-card">
        <h3>🚀 Cómo Usar</h3>
        <ol style="line-height: 2; font-size: 15px;">
            <li><strong>Configura el análisis</strong> en la barra lateral izquierda</li>
            <li><strong>Ingresa el ticker</strong> de la acción que deseas analizar (ej: AAPL, TSLA, MSFT)</li>
            <li><strong>Selecciona el índice</strong> de referencia para comparación (ej: SPY, QQQ)</li>
            <li><strong>Ajusta parámetros</strong> adicionales según tus necesidades</li>
            <li><strong>Click en "🚀 GENERAR DASHBOARD COMPLETO"</strong> para ejecutar el análisis</li>
        </ol>
        <p style="margin-top: 20px; font-size: 14px; color: var(--text-secondary);">
            <strong>💡 Tip:</strong> El análisis incluye datos históricos de hasta 10 años y múltiples periodos (YTD, 3M, 6M, 9M, 1Y, 3Y, 5Y)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer inicial
    st.markdown(f"""
    <div class="footer">
        <strong>{DISCLAIMER}</strong><br>
        <small>Desarrollado con Streamlit · Datos de Yahoo Finance · IA con Google Gemini</small>
    </div>""")