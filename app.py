# app.py
# ============================================
# Robot TCR GTQ/USD - Banguat + Pronóstico
# Autor: Henry Guzmán (para curso Modelación y Simulación)
# ============================================

import io
import re
import sys
import time
import math
import json
import hashlib
import datetime as dt
from functools import lru_cache

import requests
import numpy as np
import pandas as pd
import streamlit as st

from bs4 import BeautifulSoup
from lxml import etree
from dateutil.relativedelta import relativedelta

# Modelos
USE_PROPHET = True
try:
    from prophet import Prophet
except Exception:
    USE_PROPHET = False

USE_ARIMA = True
try:
    import statsmodels.api as sm
except Exception:
    USE_ARIMA = False


# ---------------------------
# Configuración Streamlit UI
# ---------------------------
st.set_page_config(
    page_title="Robot TCR GTQ/USD - Banguat + Pronóstico",
    page_icon="💱",
    layout="wide"
)

st.markdown(
    """
    <style>
    .ok-badge {background:#10b981; color:white; padding:3px 8px; border-radius:8px; font-size:12px;}
    .warn-badge {background:#f59e0b; color:white; padding:3px 8px; border-radius:8px; font-size:12px;}
    .err-badge {background:#ef4444; color:white; padding:3px 8px; border-radius:8px; font-size:12px;}
    .muted {color:#6b7280; font-size:12px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💱 Robot del Tipo de Cambio (GTQ/USD) — Banguat + Pronóstico")
st.caption("Ingresa una fecha y obtén el **Tipo de Cambio de Referencia** (histórico) o un **pronóstico** si la fecha es futura. Datos oficiales desde 1998.")


# ---------------------------
# Utilidades de Red
# ---------------------------
BANGUAT_SOAP_URL = "https://www.banguat.gob.gt/variables/ws/TipoCambio.asmx"
SOAP_HEADERS = {
    "Content-Type": "text/xml; charset=utf-8"
}

def _soap_envelope_fecha_inicial(fecha_ini_str: str) -> str:
    # fecha_ini_str formato dd/mm/aaaa
    return f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xmlns:xsd="http://www.w3.org/2001/XMLSchema"
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <TipoCambioFechaInicial xmlns="http://www.banguat.gob.gt/variables/ws/">
      <fecha_ini>{fecha_ini_str}</fecha_ini>
    </TipoCambioFechaInicial>
  </soap:Body>
</soap:Envelope>""".strip()


def _soap_post(body_xml: str, soap_action: str, timeout=30):
    headers = SOAP_HEADERS.copy()
    headers["SOAPAction"] = f"http://www.banguat.gob.gt/variables/ws/{soap_action}"
    resp = requests.post(BANGUAT_SOAP_URL, data=body_xml.encode("utf-8"), headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _parse_fecha_referencia(xml_bytes: bytes) -> pd.DataFrame:
    """
    Extrae pares (fecha, referencia) de cualquier respuesta SOAP del WS de Banguat
    que contenga nodos Var/VarDolar con hijos <fecha> y <referencia>.
    """
    root = etree.fromstring(xml_bytes)
    # Nodos que suelen contener la serie
    nodes = root.xpath('//*[local-name()="Var" or local-name()="VarDolar"]')
    fechas, refs = [], []
    for n in nodes:
        fecha_nodes = n.xpath('./*[local-name()="fecha"]/text()')
        ref_nodes   = n.xpath('./*[local-name()="referencia"]/text()')
        if not fecha_nodes or not ref_nodes:
            continue
        fechas.append(fecha_nodes[0])
        refs.append(ref_nodes[0])

    if not fechas:
        # Intento alterno: todas las parejas <fecha>, <referencia>
        all_fecha = root.xpath('//*[local-name()="fecha"]/text()')
        all_ref   = root.xpath('//*[local-name()="referencia"]/text()')
        if all_fecha and all_ref and len(all_fecha) == len(all_ref):
            fechas, refs = all_fecha, all_ref

    # Construir DataFrame
    if not fechas:
        return pd.DataFrame(columns=["date", "tcr"])

    # Formato fechas dd/mm/aaaa
    def parse_date(s):
        try:
            return dt.datetime.strptime(s.strip(), "%d/%m/%Y").date()
        except Exception:
            return None

    date_list = [parse_date(s) for s in fechas]
    tcr_list  = []
    for s in refs:
        s2 = s.strip().replace(",", ".")
        try:
            tcr_list.append(float(s2))
        except Exception:
            tcr_list.append(np.nan)

    df = pd.DataFrame({"date": date_list, "tcr": tcr_list})
    df = df.dropna(subset=["date", "tcr"]).sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(ttl=60*60*12)  # 12 horas
def fetch_banguat_series(start_date: dt.date = dt.date(1998, 1, 1)) -> pd.DataFrame:
    """
    Descarga serie histórica desde 'start_date' hasta hoy, usando SOAP TipoCambioFechaInicial.
    """
    fecha_ini = start_date.strftime("%d/%m/%Y")
    body = _soap_envelope_fecha_inicial(fecha_ini)
    try:
        xml_bytes = _soap_post(body, "TipoCambioFechaInicial", timeout=40)
        df = _parse_fecha_referencia(xml_bytes)
    except Exception as e:
        st.warning(f"Fallo al consultar WebService SOAP: {e}")
        df = pd.DataFrame(columns=["date", "tcr"])

    # Normalizamos a frecuencia diaria (vigente incluidos fines de semana, ffill)
    if df.empty:
        return df

    df = df.drop_duplicates("date").set_index("date").sort_index()
    full_idx = pd.date_range(df.index.min(), dt.date.today(), freq="D").date
    s = df["tcr"].reindex(full_idx)
    s = s.ffill()  # el TCR rige y se mantiene hasta el siguiente hábil
    df2 = pd.DataFrame({"date": full_idx, "tcr": s.values})
    return df2


def scrape_vigente_from_web() -> float | None:
    """
    Web Scraping del TCR vigente desde la página pública.
    Útil como verificación rápida / fallback.
    """
    try:
        url = "https://www.banguat.gob.gt/tipo_cambio/"
        html = requests.get(url, timeout=20).text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        m = re.search(r"\\b(\\d+\\.\\d{3,6})\\b", text)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


# ---------------------------
# Modelado y Pronóstico
# ---------------------------
def fit_prophet_model(df: pd.DataFrame):
    m = Prophet(
        growth="flat",
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=True,
        changepoint_prior_scale=0.3,
        interval_width=0.8
    )
    train = df.rename(columns={"date": "ds", "tcr": "y"})
    m.fit(train)
    return m

def forecast_prophet(model, target_date: dt.date, last_hist_date: dt.date) -> tuple[float, float, float]:
    days_fwd = (target_date - last_hist_date).days
    if days_fwd < 0:
        days_fwd = 0
    future = model.make_future_dataframe(periods=days_fwd, freq="D", include_history=True)
    fcst = model.predict(future)
    row = fcst.loc[fcst["ds"] == pd.Timestamp(target_date)]
    if row.empty:
        row = fcst.iloc[[-1]]
    yhat  = float(row["yhat"].values[0])
    ylo   = float(row["yhat_lower"].values[0])
    yhi   = float(row["yhat_upper"].values[0])
    return yhat, ylo, yhi


def fit_arima_model(df: pd.DataFrame):
    y = df["tcr"].astype("float64")
    best = None
    best_aic = np.inf
    for p in (0,1):
        for d in (0,1):
            for q in (0,1):
                try:
                    model = sm.tsa.SARIMAX(y, order=(p,d,q), enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)
                    if res.aic < best_aic:
                        best, best_aic = res, res.aic
                except Exception:
                    continue
    return best


def forecast_arima(model, steps: int) -> tuple[float, float, float]:
    pred = model.get_forecast(steps=steps)
    mean = float(pred.predicted_mean.iloc[-1])
    conf = pred.conf_int(alpha=0.2).iloc[-1]  # 80%
    return mean, float(conf[0]), float(conf[1])


def random_walk_forecast(last_value: float) -> tuple[float, float, float]:
    return last_value, last_value - 0.05, last_value + 0.05


# ---------------------------
# Carga de datos y UI
# ---------------------------
with st.spinner("Descargando serie oficial del WebService de Banguat (1998→hoy)..."):
    hist = fetch_banguat_series(dt.date(1998, 1, 1))

if hist.empty:
    st.error("No fue posible obtener la serie histórica desde el WS de Banguat. Intenta más tarde o verifica tu red.")
    st.stop()

# Validación cruzada rápida con scraping del vigente
vigente_scraped = scrape_vigente_from_web()
col_a, col_b, col_c = st.columns([2,1,1])
with col_a:
    st.subheader("Serie histórica oficial (Banguat)")
with col_b:
    st.metric("Último día en la serie", value=str(hist["date"].max()))
with col_c:
    st.metric("Valor vigente (WS)", value=f"Q {hist.set_index('date').loc[hist['date'].max(), 'tcr']:.5f}")

if vigente_scraped is not None:
    v_ws = hist.set_index("date").loc[hist["date"].max(), "tcr"]
    delta = vigente_scraped - v_ws
    badge = '<span class="ok-badge">OK</span>' if abs(delta) < 0.01 else '<span class="warn-badge">Difiere</span>'
    st.markdown(f"**Scraping del vigente:** Q {vigente_scraped:.5f} {badge} <span class='muted'>(comparado con WS)</span>", unsafe_allow_html=True)


# Panel de control
st.markdown("---")
left, right = st.columns([1,1])
with left:
    target_date = st.date_input(
        "Selecciona una fecha (pasada o futura):",
        value=dt.date.today(),
        min_value=hist["date"].min(),
        max_value=dt.date.today() + relativedelta(years=5),
        help="Para pasado, se devuelve el dato oficial; para futuro, un pronóstico con intervalo de confianza."
    )
with right:
    model_choice = st.radio(
        "Modelo de pronóstico (si la fecha es futura):",
        options=["Prophet (recomendado)", "ARIMA (alterno)", "Random Walk (baseline)"],
        horizontal=True
    )

last_hist_date = hist["date"].max()
tcr_today = hist.loc[hist["date"] == last_hist_date, "tcr"].values[0]
is_future = target_date > last_hist_date

# Bloque de resultado
st.markdown("### Resultado")

if not is_future:
    val = hist.set_index("date").loc[target_date, "tcr"]
    st.success(f"**Tipo de Cambio de Referencia (oficial Banguat) para {target_date}: Q {val:.5f} por USD**")
    st.caption("Fuente: WebService oficial de Banguat. El TCR rige el día hábil siguiente al cálculo y se mantiene vigente hasta su actualización.")
else:
    horizon_days = (target_date - last_hist_date).days

    yhat, lo, hi = None, None, None
    model_used = None

    try:
        if model_choice.startswith("Prophet") and USE_PROPHET:
            model = fit_prophet_model(hist)
            yhat, lo, hi = forecast_prophet(model, target_date, last_hist_date)
            model_used = "Prophet"
        elif model_choice.startswith("ARIMA") and USE_ARIMA:
            arima = fit_arima_model(hist)
            yhat, lo, hi = forecast_arima(arima, steps=horizon_days)
            model_used = "ARIMA"
        else:
            yhat, lo, hi = random_walk_forecast(tcr_today)
            model_used = "Random Walk"
    except Exception as e:
        try:
            if USE_ARIMA:
                arima = fit_arima_model(hist)
                yhat, lo, hi = forecast_arima(arima, steps=horizon_days)
                model_used = "ARIMA"
            else:
                yhat, lo, hi = random_walk_forecast(tcr_today)
                model_used = "Random Walk"
        except Exception:
            yhat, lo, hi = random_walk_forecast(tcr_today)
            model_used = "Random Walk"

    st.info(
        f"**Pronóstico del TCR** para **{target_date}** "
        f"(modelo: **{model_used}**): **Q {yhat:.5f}** por USD "
        f"(_IC 80%: {lo:.5f} ⟷ {hi:.5f}_)"
    )
    st.caption("Advertencia: este valor es un pronóstico estadístico; puede diferir del TCR oficial que publique Banguat en esa fecha.")

# Gráfica
st.markdown("---")
st.subheader("Histórico y (opcional) pronóstico cercano")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(pd.to_datetime(hist["date"]), hist["tcr"], linewidth=1.5, label="Histórico (WS Banguat)")
ax.set_xlabel("Fecha")
ax.set_ylabel("Q por USD")
ax.grid(True, alpha=0.25)

# Pronóstico corto si es futuro
if is_future:
    try:
        if model_choice.startswith("Prophet") and USE_PROPHET:
            model = fit_prophet_model(hist)
            future = model.make_future_dataframe(periods=30, freq="D", include_history=True)
            fcst = model.predict(future)
            mask = fcst["ds"] >= (pd.Timestamp(hist["date"].max()) - pd.Timedelta(days=120))
            ax.plot(fcst.loc[mask, "ds"], fcst.loc[mask, "yhat"], linestyle="--", linewidth=1.2, label="Pronóstico (corto)")
        elif model_choice.startswith("ARIMA") and USE_ARIMA:
            steps = 30
            arima = fit_arima_model(hist)
            pred = arima.get_forecast(steps=steps)
            idx = pd.date_range(hist["date"].max() + dt.timedelta(days=1), periods=steps, freq="D")
            ax.plot(idx, pred.predicted_mean.values, linestyle="--", linewidth=1.2, label="Pronóstico (corto)")
        else:
            idx = pd.date_range(hist["date"].max() + dt.timedelta(days=1), periods=30, freq="D")
            ax.plot(idx, np.full(len(idx), tcr_today), linestyle="--", linewidth=1.2, label="Pronóstico (corto)")
    except Exception:
        pass

ax.legend(loc="best")
st.pyplot(fig, clear_figure=True)

# Descargar CSV
csv_bytes = hist.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar histórico (CSV)",
    data=csv_bytes,
    file_name="banguat_tcr_gtq_usd.csv",
    mime="text/csv"
)

# Notas
with st.expander("Notas importantes / Metodología"):
    st.markdown(
        """
- **Fuente primaria:** WebService SOAP del Banco de Guatemala. Se consume el método `TipoCambioFechaInicial` desde 01/01/1998 hasta hoy y se normaliza a frecuencia diaria con *forward-fill* (el TCR rige hasta la próxima actualización).
- **Scraping de apoyo:** Se consulta la página pública del TCR vigente para verificación rápida; si difiere ±0.01 puede deberse a redondeos o tiempos de publicación.
- **Pronóstico:** Por defecto **Prophet** (modelo aditivo con cambios de tendencia y estacionalidad anual). Alternativas: **ARIMA** y **Random Walk** (baseline).
- **Intervalos:** IC 80% para una comunicación conservadora.
- **Limitación:** El TCR tiende a comportarse como *random walk*; pronósticos a largo plazo convergen al último valor (amplios intervalos).
- **Uso académico:** Este robot sirve para explorar escenarios; para usos normativos/tributarios consulte siempre el TCR oficial publicado por Banguat.
        """
    )
