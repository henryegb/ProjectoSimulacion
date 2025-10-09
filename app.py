# app.py (offline-first)
# Robot TCR GTQ/USD — Banguat + Pronóstico
# Carga primero data/banguat_tcr_gtq_usd.csv (versión controlada en GitHub).
# Si no existe, intenta descargar via WS en memoria (sin persistir).

import os
import datetime as dt

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import numpy as np
import pandas as pd
import streamlit as st
from lxml import etree
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

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
st.title("💱 Robot del Tipo de Cambio (GTQ/USD) — Banguat + Pronóstico (offline-first)")
st.caption("Usa el CSV del repositorio como fuente principal. WS sólo como respaldo.")

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
    resp.raise_for_status()
    return resp.content

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
        if all_fecha and all_ref and len(all_fecha)==len(all_ref): 
            fechas, refs = all_fecha, all_ref
    if not fechas:
        return pd.DataFrame(columns=["date","tcr"])
    def parse_date(s):
        try: 
            return pd.to_datetime(s, format="%d/%m/%Y").date()
        except: 
            return None
    date_list = [parse_date(s) for s in fechas]
    tcr_list = []
    for s in refs:
        try: 
            tcr_list.append(float(str(s).replace(",",".")))
        except: 
            tcr_list.append(np.nan)
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
                if not tmp.empty: 
                    got = tmp
                    break
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
    # — HISTÓRICO —
    val = hist.set_index("date").loc[sel_date, "tcr"]
    st.success(f"**TCR oficial para {sel_date}: Q {val:.5f}**")

else:
    # — PRONÓSTICO —
    y = hist["tcr"].astype("float64")
    horizon = (sel_date - last_date).days
    try:
        if model_choice.startswith("ARIMA") and USE_ARIMA:
            model = sm.tsa.SARIMAX(y, order=(1,1,0),
                                   enforce_stationarity=False,
                                   enforce_invertibility=False).fit(disp=False)
            pred = model.get_forecast(steps=horizon)
            yhat = float(pred.predicted_mean.iloc[-1])
            lo, hi = pred.conf_int(alpha=0.2).iloc[-1]
            model_used = "ARIMA(1,1,0)"
        elif model_choice.startswith("Prophet") and USE_PROPHET:
            m = Prophet(growth="flat", daily_seasonality=True,
                        weekly_seasonality=False, yearly_seasonality=True,
                        interval_width=0.8)
            m.fit(hist.rename(columns={"date":"ds","tcr":"y"}))
            future = m.make_future_dataframe(periods=horizon, freq="D", include_history=True)
            fcst = m.predict(future)
            row = fcst.loc[fcst["ds"] == pd.Timestamp(sel_date)]
            if row.empty: 
                row = fcst.iloc[[-1]]
            yhat = float(row["yhat"]); lo = float(row["yhat_lower"]); hi = float(row["yhat_upper"])
            model_used = "Prophet"
        else:
            yhat = y.iloc[-1]; lo, hi = yhat-0.05, yhat+0.05; model_used = "Random Walk"
    except Exception:
        yhat = y.iloc[-1]; lo, hi = yhat-0.05, yhat+0.05; model_used = "Random Walk"

    st.info(f"**Pronóstico para {sel_date}** (modelo **{model_used}**): "
            f"**Q {yhat:.5f}**  _IC 80%: {lo:.5f} ⟷ {hi:.5f}_")

    # === FIGURA 2: cola (24 meses) + pronóstico con banda ===
    win_months = 24
    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"])
    cut = pd.Timestamp(last_date) - pd.DateOffset(months=win_months)
    h = h[h["date"] >= cut]

    steps = max(1, (sel_date - last_date).days)
    if model_used.startswith("ARIMA") and 'model' in locals():
        pred = model.get_forecast(steps=steps)
        f_dates = pd.date_range(last_date + pd.Timedelta(days=1), sel_date, freq="D")
        f_mean = pred.predicted_mean.to_numpy()
        ci = pred.conf_int(alpha=0.2).to_numpy()
        f_lo, f_hi = ci[:, 0], ci[:, 1]

    elif model_used.startswith("Prophet") and 'fcst' in locals():
        f = fcst[fcst["ds"] > pd.Timestamp(last_date)]
        f_dates = pd.to_datetime(f["ds"])
        f_mean = f["yhat"].to_numpy()
        f_lo = f["yhat_lower"].to_numpy()
        f_hi = f["yhat_upper"].to_numpy()

    else:
        # Random Walk: línea horizontal y banda constante
        f_dates = pd.date_range(last_date + pd.Timedelta(days=1), sel_date, freq="D")
        last_val = float(h["tcr"].iloc[-1])
        f_mean = np.full(len(f_dates), last_val)
        f_lo   = np.full(len(f_dates), float(lo))
        f_hi   = np.full(len(f_dates), float(hi))

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

    st.pyplot(fig2, clear_figure=True)
    fig2.savefig("fig_pronostico.png", dpi=300, bbox_inches="tight")

# ---------- Figura 1: histórico completo ----------
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(pd.to_datetime(hist["date"]), hist["tcr"], linewidth=1.5, label="Histórico")
ax.set_xlabel("Fecha"); ax.set_ylabel("Q por USD"); ax.grid(True, alpha=0.25)
ax.legend(loc="best")
st.pyplot(fig, clear_figure=True)
fig.savefig("fig_hist.png", dpi=300, bbox_inches="tight")

# ---------- Descarga ----------
st.download_button("⬇️ Descargar CSV (histórico en uso)", data=hist.to_csv(index=False).encode("utf-8"),
                   file_name="banguat_tcr_gtq_usd.csv", mime="text/csv")

with st.expander("Notas"):
    st.markdown("- Este modo usa **CSV versionado**. Para mantenerlo actualizado, configura el **workflow de GitHub Actions** incluido en el repo.")

# =========================================================
# === BLOQUE: DESARROLLO 95% (tramos + back-testing)
# =========================================================
from statsmodels.stats.stattools import jarque_bera

st.markdown("---")
st.header("Desarrollo 95% (tramos + back-testing)")

# ---------- Helpers de modelado ----------
def fit_poly(x, y, deg):
    X = np.vstack([x**k for k in range(deg+1)]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else 0.0
    return yhat, r2, {"beta": beta}

def fit_loglin(x, y):
    y_safe = np.where(y <= 0, np.nan, y)
    mask = ~np.isnan(y_safe)
    if mask.sum() < 3:
        return np.full_like(y, y.mean(), dtype=float), 0.0, {"ok": False}
    Y = np.log(y_safe[mask])
    X = np.vstack([np.ones(mask.sum()), x[mask]]).T
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    yhat = np.exp(beta[0] + beta[1]*x)
    ss_res = float(np.nansum((y - yhat)**2))
    ss_tot = float(np.nansum((y - np.nanmean(y))**2))
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else 0.0
    return yhat, r2, {"beta": beta, "ok": True}

def moving_average(y, win=30):
    return pd.Series(y).rolling(win, min_periods=1).mean().values

def fit_ses(y):
    # SES simple por búsqueda de alpha que minimiza RMSE in-sample
    best = (np.inf, 0.2, None)  # (rmse, alpha, yhat)
    for alpha in np.linspace(0.05, 0.95, 19):
        yhat = np.zeros_like(y, dtype=float)
        yhat[0] = y[0]
        for t in range(1, len(y)):
            yhat[t] = alpha*y[t-1] + (1-alpha)*yhat[t-1]
        rmse = float(np.sqrt(np.mean((y - yhat)**2)))
        if rmse < best[0]:
            best = (rmse, alpha, yhat.copy())
    yhat = best[2]
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else 0.0
    return yhat, r2, {"alpha": best[1]}

def pearson_r(y, yhat):
    y1 = y - y.mean(); y2 = yhat - yhat.mean()
    denom = float(np.sqrt((y1**2).sum() * (y2**2).sum()))
    return float((y1*y2).sum()/denom) if denom>0 else 0.0

def metrics_test(y_true, y_pred):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / y_true)) * 100)
    rmse_pct = float(rmse / max(1e-9, np.mean(y_true)) * 100)
    jb_stat, jb_p, _, _ = jarque_bera(err)
    return mae, rmse, mape, rmse_pct, float(jb_p)

def choose_model(x_tr, y_tr, allow):
    cands = []
    if allow["poly1"]:
        y1, r2_1, p1 = fit_poly(x_tr, y_tr, 1); cands.append(("Poly1", y1, r2_1, p1))
    if allow["poly2"]:
        y2, r2_2, p2 = fit_poly(x_tr, y_tr, 2); cands.append(("Poly2", y2, r2_2, p2))
    if allow["poly3"]:
        y3, r2_3, p3 = fit_poly(x_tr, y_tr, 3); cands.append(("Poly3", y3, r2_3, p3))
    if allow["log"]:
        yl, r2_l, pl = fit_loglin(x_tr, y_tr); cands.append(("Log-lin", yl, r2_l, pl))
    if allow["ses"]:
        yes, r2_es, pes = fit_ses(y_tr); cands.append(("SES", yes, r2_es, pes))
    if allow["ma"]:
        yma = moving_average(y_tr, allow["ma_win"])
        ss_res = float(np.sum((y_tr - yma)**2)); ss_tot = float(np.sum((y_tr - y_tr.mean())**2))
        r2_ma = 1 - ss_res/ss_tot if ss_tot>0 else 0.0
        cands.append((f"MA({allow['ma_win']})", yma, r2_ma, {"k": allow["ma_win"]}))
    return cands

def predict_model(name, pars, x_te, y_tr_last):
    if name.startswith("Poly"):
        deg = int(name[-1]); beta = pars["beta"]
        Xte = np.vstack([x_te**k for k in range(deg+1)]).T
        return Xte @ beta
    if name == "Log-lin" and pars.get("ok", False):
        b0, b1 = pars["beta"]; 
        return np.exp(b0 + b1*x_te)
    # SES / MA: extensión naive (mantener último nivel) para test
    return np.full_like(x_te, y_tr_last, dtype="float64")

def build_segments_from_cuts(df, cut_str, min_days=180):
    cuts = []
    if cut_str.strip():
        for tok in cut_str.split(","):
            try:
                cuts.append(pd.to_datetime(tok.strip()).normalize())
            except Exception:
                pass
    cuts = [c for c in sorted(cuts) if df["date"].min() < c < df["date"].max()]
    segs, start = [], df["date"].min().normalize()
    for c in cuts:
        segs.append((start, c))
        start = c + pd.Timedelta(days=1)
    segs.append((start, df["date"].max().normalize()))
    segs = [s for s in segs if (s[1]-s[0]).days + 1 >= min_days]
    return segs

def evaluate_segments(df, segments, allow, target="r"):
    rows = []
    w_sum = 0.0; r_num = 0.0; r2_num = 0.0
    for (a,b) in segments:
        d = df[(df["date"]>=a) & (df["date"]<=b)].copy().reset_index(drop=True)
        n = len(d)
        if n < 20:
            continue
        n_tr = max(int(n*0.8), 1)
        d_tr, d_te = d.iloc[:n_tr], d.iloc[n_tr:]
        if len(d_te)==0:
            continue
        x0 = d_tr["date"].iloc[0]
        x_tr = (d_tr["date"] - x0).dt.days.values.astype(float)
        y_tr = d_tr["tcr"].astype("float64").values
        x_te = (d_te["date"] - x0).dt.days.values.astype(float)
        y_te = d_te["tcr"].astype("float64").values

        all_cands = choose_model(x_tr, y_tr, allow)
        scored = []
        for name, yhat_tr, r2_tr, pars in all_cands:
            r_tr = pearson_r(y_tr, yhat_tr)
            score = r_tr if target == "r" else r2_tr
            scored.append((score, name, yhat_tr, r2_tr, r_tr, pars))
        if not scored:
            continue
        scored.sort(key=lambda z: z[0], reverse=True)
        _, model, yhat_tr, r2_tr, r_tr, pars = scored[0]

        yhat_te = predict_model(model, pars, x_te, y_tr[-1])
        mae, rmse, mape, rmse_pct, jb_p = metrics_test(y_te, yhat_te)

        rows.append(dict(
            tramo=f"{a.date()}–{b.date()}",
            n=int(n),
            modelo=model,
            R2_train=round(r2_tr*100,2),
            r_train=round(r_tr*100,2),
            MAE=round(mae,5),
            RMSE=round(rmse,5),
            MAPE=round(mape,3),
            RMSEpct=round(rmse_pct,3),
            JB_p=round(jb_p,4),
        ))

        w = n
        w_sum += w
        r2_num += (r2_tr*100)*w
        r_num  += (r_tr*100)*w

    tabla = pd.DataFrame(rows)
    R2_prom = float(r2_num/w_sum) if w_sum>0 else float("nan")
    r_prom  = float(r_num /w_sum) if w_sum>0 else float("nan")
    return tabla, R2_prom, r_prom

# ---------- UI ----------
col = st.columns([2,1,1,1])
with col[0]:
    default_cuts = "2004-12-31, 2008-12-31, 2016-12-31, 2019-12-31, 2023-12-31"
    cut_str = st.text_input("Fechas de corte (YYYY-MM-DD, separadas por coma)", value=default_cuts)
with col[1]:
    min_days = st.number_input("Mín. días por tramo", min_value=120, max_value=2000, value=180, step=30)
with col[2]:
    target_metric = st.selectbox("Meta de ajuste", ["r (correlación)", "R2 (determinación)"], index=0)
with col[3]:
    ma_win = st.number_input("Ventana MA (días)", min_value=7, max_value=90, value=30, step=1)

c2 = st.columns(6)
with c2[0]: allow_poly1 = st.checkbox("Poly1", True)
with c2[1]: allow_poly2 = st.checkbox("Poly2", True)
with c2[2]: allow_poly3 = st.checkbox("Poly3", False)
with c2[3]: allow_log   = st.checkbox("Log-lin", True)
with c2[4]: allow_ses   = st.checkbox("SES", True)
with c2[5]: allow_ma    = st.checkbox("MA", True)

if st.button("Calcular tramos y métricas", type="primary"):
    df95 = hist.copy()
    df95["date"] = pd.to_datetime(df95["date"])
    df95 = df95.sort_values("date").reset_index(drop=True)

    allow = dict(poly1=allow_poly1, poly2=allow_poly2, poly3=allow_poly3,
                 log=allow_log, ses=allow_ses, ma=allow_ma, ma_win=int(ma_win))
    segments = build_segments_from_cuts(df95, cut_str, min_days=min_days)

    if not segments:
        st.warning("No se construyeron tramos (revisa cortes y/o mínimo de días).")
    else:
        st.write("**Tramos:**", " | ".join([f"{a.date()}–{b.date()}" for a,b in segments]))
        tabla, R2_prom, r_prom = evaluate_segments(
            df95, segments, allow,
            target=("r" if target_metric.startswith("r") else "R2")
        )

        if tabla.empty:
            st.warning("No hubo suficiente data en los tramos para back-testing.")
        else:
            cmet = st.columns(3)
            with cmet[0]:
                st.metric("r promedio ponderado (entrenamiento)", f"{r_prom:.2f}%")
            with cmet[1]:
                st.metric("R² promedio ponderado (entrenamiento)", f"{R2_prom:.2f}%")
            with cmet[2]:
                bad_rmse = (tabla["RMSEpct"] > 5.0).sum()
                st.metric("Tramos con RMSE% > 5", f"{bad_rmse}/{len(tabla)}")

            jb_ok_pct = (tabla["JB_p"] > 0.05).mean() * 100.0
            st.caption(f"Prueba de normalidad (JB): {jb_ok_pct:.1f}% de los tramos con p>0.05 (no rechazo H0).")

            # objetivo de consigna
            objetivo_ok = (r_prom >= 95.0)
            st.info("Objetivo sugerido por consigna: **coeficiente de correlación (r) ≥ 95%** (promedio ponderado), "
                    "**RMSE% ≤ 5** por tramo y residuos con **JB p>0.05**.")
            if objetivo_ok and bad_rmse == 0:
                st.success("✅ Meta alcanzada (r ≥ 95% y RMSE% ≤ 5 en todos los tramos).")
            elif objetivo_ok:
                st.warning("🟡 r ≥ 95%, pero hay tramos con RMSE% > 5. Ajusta cortes/modelos.")
            else:
                st.warning("⚠️ r promedio < 95%. Prueba otros cortes o activa Poly3/Log-lin/SES/MA.")

            st.dataframe(tabla, use_container_width=True)

            # Descargas
            st.download_button("⬇️ CSV (tramos + métricas)", data=tabla.to_csv(index=False).encode("utf-8"),
                               file_name="tramos_backtesting.csv", mime="text/csv")

            # Bloque LaTeX (CORREGIDO SIN SALTOS EN F-STRING)
            lines = []
            for _, r in tabla.iterrows():
                line = (
                    f"{r['tramo']} & {r['n']} & {r['modelo']} & {r['R2_train']:.2f} & "
                    f"{r['r_train']:.2f} & {r['MAE']:.5f} & {r['RMSE']:.5f} & "
                    f"{r['MAPE']:.2f} & {r['RMSEpct']:.2f} & {r['JB_p']:.3f} \\\\"
                )
                lines.append(line)

            latex_block = (
                "\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
                "Tramo & n & Modelo & R\\textsuperscript{2} (\\%) & r (\\%) & MAE & RMSE & MAPE & RMSE\\% & JB p \\\\\n"
                "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
            )
            st.download_button("⬇️ Bloque LaTeX (tabla de tramos)",
                               data=latex_block.encode("utf-8"),
                               file_name="tabla_tramos_latex.tex",
                               mime="text/plain")
