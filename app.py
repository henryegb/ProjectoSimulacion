# app.py (offline-first)
# Robot TCR GTQ/USD — Banguat + Pronóstico
# Carga primero data/banguat_tcr_gtq_usd.csv (versión controlada en GitHub).
# Si no existe, intenta descargar via WS en memoria (sin persistir).

import os
import re
import datetime as dt

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from lxml import etree
from dateutil.relativedelta import relativedelta

# ----- Modelos -----
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

st.set_page_config(page_title="Robot TCR GTQ/USD — Offline-first", page_icon="💱", layout="wide")
st.title("💱 Robot del Tipo de Cambio (GTQ/USD) — Banguat + Pronóstico (offline‑first)")
st.caption("Usa el CSV del repositorio como fuente principal. WS solo como respaldo.")

# ---------- UI (sidebar) ----------
with st.sidebar:
    st.header("Consulta")
    sel_date = st.date_input("Selecciona una fecha:", value=dt.date.today(),
                             min_value=dt.date(1998,1,1),
                             max_value=dt.date.today() + relativedelta(years=5))
    model_choice = st.radio("Modelo (si es futuro):",
                            ["ARIMA (alterno)", "Random Walk (baseline)"] + (["Prophet (opcional)"] if USE_PROPHET else []),
                            index=0)
    st.divider()
    st.subheader("Datos")
    st.write("Fuente prioritaria: `data/banguat_tcr_gtq_usd.csv` (en el repo).")
    try_ws = st.checkbox("Si no hay CSV, intentar WS de Banguat", value=True)

# ---------- Utilidades ----------
def _requests_session():
    s = requests.Session()
    retry = Retry(total=5, connect=5, read=5, backoff_factor=0.5,
                  status_forcelist=[500,502,503,504], allowed_methods=["GET","POST"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

SESSION = _requests_session()
BANGUAT_SOAP_URL = "https://www.banguat.gob.gt/variables/ws/TipoCambio.asmx"
SOAP_HEADERS = {"Content-Type":"text/xml; charset=utf-8"}

def _soap_post(body_xml: str, soap_action: str, timeout=30):
    headers = SOAP_HEADERS.copy()
    headers["SOAPAction"] = f"http://www.banguat.gob.gt/variables/ws/{soap_action}"
    resp = SESSION.post(BANGUAT_SOAP_URL, data=body_xml.encode("utf-8"), headers=headers, timeout=timeout)
    resp.raise_for_status(); return resp.content

def _soap_envelope_rango(fini:str, ffin:str, variant:int=0) -> str:
    if variant == 0:
        params = f"<fechainit>{fini}</fechainit><fechafin>{ffin}</fechafin>"
    elif variant == 1:
        params = f"<fecha_ini>{fini}</fecha_ini><fecha_fin>{ffin}</fecha_fin>"
    else:
        params = f"<fechaini>{fini}</fechaini><fechafin>{ffin}</fechafin>"
    return f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xmlns:xsd="http://www.w3.org/2001/XMLSchema"
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <TipoCambioRango xmlns="http://www.banguat.gob.gt/variables/ws/">
      {params}
    </TipoCambioRango>
  </soap:Body>
</soap:Envelope>""".strip()

def _parse_fecha_referencia(xml_bytes: bytes) -> pd.DataFrame:
    root = etree.fromstring(xml_bytes)
    nodes = root.xpath('//*[local-name()="Var" or local-name()="VarDolar"]')
    fechas, refs = [], []
    for n in nodes:
        fecha_nodes = n.xpath('./*[local-name()="fecha"]/text()')
        ref_nodes   = n.xpath('./*[local-name()="referencia"]/text()')
        if fecha_nodes and ref_nodes:
            fechas.append(fecha_nodes[0]); refs.append(ref_nodes[0])
    if not fechas:
        all_fecha = root.xpath('//*[local-name()="fecha"]/text()')
        all_ref   = root.xpath('//*[local-name()="referencia"]/text()')
        if all_fecha and all_ref and len(all_fecha)==len(all_ref): fechas, refs = all_fecha, all_ref
    if not fechas: return pd.DataFrame(columns=["date","tcr"])
    def parse_date(s):
        try: return pd.to_datetime(s, format="%d/%m/%Y").date()
        except: return None
    date_list = [parse_date(s) for s in fechas]
    tcr_list = []
    for s in refs:
        try: tcr_list.append(float(str(s).replace(",",".")))
        except: tcr_list.append(np.nan)
    df = pd.DataFrame({"date":date_list,"tcr":tcr_list}).dropna().sort_values("date").reset_index(drop=True)
    return df

def _normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates("date").set_index("date").sort_index()
    full_idx = pd.date_range(df.index.min(), dt.date.today(), freq="D").date
    s = df["tcr"].reindex(full_idx).ffill()
    return pd.DataFrame({"date": full_idx, "tcr": s.values})

@st.cache_data(ttl=60*60*12)
def fetch_ws_memory(start_date=dt.date(1998,1,1)) -> pd.DataFrame:
    parts = []
    fini = start_date; ffin = dt.date.today()
    step = dt.timedelta(days=120)
    while fini <= ffin:
        end = min(fini + step, ffin)
        fini_s, end_s = fini.strftime("%d/%m/%Y"), end.strftime("%d/%m/%Y")
        got = None
        for variant in (0,1,2):
            try:
                body = _soap_envelope_rango(fini_s, end_s, variant)
                xml_bytes = _soap_post(body, "TipoCambioRango", timeout=40)
                tmp = _parse_fecha_referencia(xml_bytes)
                if not tmp.empty: got = tmp; break
            except Exception:
                continue
        if got is not None:
            parts.append(got)
        fini = end + dt.timedelta(days=1)
    if parts:
        df = pd.concat(parts, ignore_index=True)
        df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        return _normalize_daily(df)
    return pd.DataFrame(columns=["date","tcr"])

# ---------- Carga de datos (CSV primero) ----------
hist = None
csv_path = os.path.join("data","banguat_tcr_gtq_usd.csv")
if os.path.exists(csv_path):
    hist = pd.read_csv(csv_path)
    hist["date"] = pd.to_datetime(hist["date"]).dt.date
    hist = _normalize_daily(hist)
    source_label = "CSV del repositorio"
else:
    if try_ws:
        with st.spinner("No hay CSV. Intentando WS de Banguat (modo memoria)..."):
            hist = fetch_ws_memory(dt.date(1998,1,1))
        source_label = "WS (memoria)"
    else:
        hist = pd.DataFrame(columns=["date","tcr"])
        source_label = "Sin datos"

if hist.empty:
    st.error("No hay datos. Agrega `data/banguat_tcr_gtq_usd.csv` al repositorio o activa WS.")
    st.stop()

# ---------- Resultado ----------
last_date = hist["date"].max()
is_future = sel_date > last_date

colA, colB, colC = st.columns([2,1,1])
with colA: st.markdown(f"**Fuente:** {source_label}")
with colB: st.metric("Último día con dato", str(last_date))
with colC: st.metric("Observaciones", f"{len(hist):,}")

st.markdown("### Resultado")
if not is_future:
    val = hist.set_index("date").loc[sel_date, "tcr"]
    st.success(f"**TCR oficial para {sel_date}: Q {val:.5f}**")
else:
    # modelos simples
    y = hist["tcr"].astype("float64")
    horizon = (sel_date - last_date).days
    try:
        if model_choice.startswith("ARIMA") and USE_ARIMA:
            model = sm.tsa.SARIMAX(y, order=(1,1,0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            pred = model.get_forecast(steps=horizon)
            yhat = float(pred.predicted_mean.iloc[-1])
            lo, hi = pred.conf_int(alpha=0.2).iloc[-1]
            model_used = "ARIMA(1,1,0)"
        elif model_choice.startswith("Prophet") and USE_PROPHET:
            m = Prophet(growth="flat", daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=True, interval_width=0.8)
            m.fit(hist.rename(columns={"date":"ds","tcr":"y"}))
            future = m.make_future_dataframe(periods=horizon, freq="D", include_history=True)
            fcst = m.predict(future)
            row = fcst.loc[fcst["ds"] == pd.Timestamp(sel_date)]
            if row.empty: row = fcst.iloc[[-1]]
            yhat = float(row["yhat"]); lo = float(row["yhat_lower"]); hi = float(row["yhat_upper"])
            model_used = "Prophet"
        else:
            yhat = y.iloc[-1]; lo, hi = yhat-0.05, yhat+0.05; model_used = "Random Walk"
    except Exception:
        yhat = y.iloc[-1]; lo, hi = yhat-0.05, yhat+0.05; model_used = "Random Walk"

    st.info(f"**Pronóstico para {sel_date}** (modelo **{model_used}**): **Q {yhat:.5f}**  _IC 80%: {lo:.5f} ⟷ {hi:.5f}_")
# === FIGURA 2: cola (24 meses) + pronóstico con banda ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

win_months = 24
h = hist.copy()
h["date"] = pd.to_datetime(h["date"])
cut = pd.Timestamp(last_date) - pd.DateOffset(months=win_months)
h = h[h["date"] >= cut]

# Trayectoria futura según el modelo elegido arriba
steps = max(1, (sel_date - last_date).days)  # por si el horizonte es 0
if model_used.startswith("ARIMA") and 'model' in locals():
    pred = model.get_forecast(steps=steps)
    f_dates = pd.date_range(last_date + pd.Timedelta(days=1), sel_date, freq="D")
    f_mean = pred.predicted_mean.to_numpy()
    ci = pred.conf_int(alpha=0.2).to_numpy()  # 80%
    f_lo, f_hi = ci[:, 0], ci[:, 1]

elif model_used.startswith("Prophet") and 'fcst' in locals():
    f = fcst[fcst["ds"] > pd.Timestamp(last_date)]
    f_dates = pd.to_datetime(f["ds"])
    f_mean = f["yhat"].to_numpy()
    f_lo = f["yhat_lower"].to_numpy()
    f_hi = f["yhat_upper"].to_numpy()

else:
    # Random Walk: línea horizontal (último valor) y banda constante
    f_dates = pd.date_range(last_date + pd.Timedelta(days=1), sel_date, freq="D")
    last_val = float(h["tcr"].iloc[-1])
    f_mean = np.full(len(f_dates), last_val)
    f_lo = np.full(len(f_dates), float(lo))
    f_hi = np.full(len(f_dates), float(hi))

fig2, ax2 = plt.subplots(figsize=(8,3))
ax2.plot(h["date"], h["tcr"], label="Histórico", linewidth=1.4)
ax2.plot(f_dates, f_mean, label="Pronóstico", linewidth=1.6)
ax2.fill_between(f_dates, f_lo, f_hi, alpha=0.20, label="IC 80%")
ax2.axvline(pd.Timestamp(last_date), linestyle="--", linewidth=1, alpha=0.6)
ax2.scatter([pd.Timestamp(sel_date)],
            [f_mean[-1] if len(f_mean) else float(yhat)],
            s=30, zorder=3, label=f"{sel_date}")
ax2.set_xlabel("Fecha"); ax2.set_ylabel("Q por USD")
ax2.grid(True, alpha=0.25); ax2.legend(loc="best")

# Muestra en la app y guarda PNG para el paper
st.pyplot(fig2, clear_figure=True)
fig2.savefig("fig_pronostico.png", dpi=300, bbox_inches="tight")

# ---------- Gráfico ----------
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(pd.to_datetime(hist["date"]), hist["tcr"], linewidth=1.5, label="Histórico")
ax.set_xlabel("Fecha"); ax.set_ylabel("Q por USD"); ax.grid(True, alpha=0.25)
ax.legend(loc="best")
st.pyplot(fig, clear_figure=True)

# ---------- Descarga ----------
st.download_button("⬇️ Descargar CSV (histórico en uso)", data=hist.to_csv(index=False).encode("utf-8"),
                   file_name="banguat_tcr_gtq_usd.csv", mime="text/csv")

with st.expander("Notas"):
    st.markdown("- Este modo usa **CSV versionado**. Para mantenerlo actualizado, configura el **workflow de GitHub Actions** incluido en el repo.")
