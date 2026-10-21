import numpy as np, pandas as pd, streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="Fuel Hedge What-If", layout="wide")
RAW = {2020:[1.770,1.509,0.953,0.606,0.686,0.983,1.084,1.112,1.006,1.047,1.133,1.324],
2021:[1.420,1.601,1.661,1.669,1.752,1.857,1.887,1.817,2.003,2.296,2.193,2.095],
2022:[2.446,2.682,3.495,3.914,3.902,4.120,3.478,3.339,3.262,3.715,3.160,2.897],
2023:[3.542,2.790,2.679,2.374,2.173,2.250,2.488,2.989,3.120,2.881,2.734,2.387],
2024:[2.585,2.683,2.593,2.605,2.380,2.394,2.419,2.176,1.982,2.068,2.065,2.097],
2025:[2.347,2.264,2.069,1.976,1.928,2.094,2.244,2.024,2.089]}

@st.cache_data
def default_prices():
    s = {pd.Timestamp(y, m+1, 1): v*42 for y, l in RAW.items() for m, v in enumerate(l)}
    return pd.Series(s).sort_index()

def load_prices(up):
    if up is None:
        return default_prices()
    df = pd.read_csv(up)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0]).dt.to_period("M").dt.to_timestamp()
    return df.set_index(df.columns[0]).iloc[:, 0].astype(float).sort_index()

def b76(F, K, T, s):
    d1 = (np.log(F/K) + .5*s*s*T) / (s*np.sqrt(T)); d2 = d1 - s*np.sqrt(T)
    return F*norm.cdf(d1) - K*norm.cdf(d2)

def run(px, budget, otm, vol_mode, vol, annual, maxh, tmin, tmax, floor):
    rv = np.log(px).diff().rolling(12, min_periods=6).std()*np.sqrt(12)
    tenors = list(range(tmin, tmax+1)); layer = maxh*(annual/12)/len(tenors)
    spent, out = {}, []
    for t in px.index:
        v = vol if vol_mode == "Flat" else max(rv.get(t, np.nan) if not np.isnan(rv.get(t, np.nan)) else vol, floor)
        for h in tenors:
            e = t + pd.DateOffset(months=h)
            if e not in px.index: continue
            S = px[t]; K = S*(1+otm); p = b76(S, K, h/12, v)
            used = spent.get(t.year, 0); q = layer
            if used + p*q > budget: q = max(0, (budget-used)/p)
            spent[t.year] = used + p*q
            out.append(dict(buy=t, expiry=e, tenor=h, vol=v, spot=S, strike=K, prem_bbl=p,
                            bbl=q, premium=p*q, payout=max(px[e]-K, 0)*q))
    return pd.DataFrame(out)

st.title("Jet fuel call-option hedge: what if we had hedged since 2020?")
with st.sidebar:
    budget = st.number_input("Annual premium budget (USD m)", 0.5, 50.0, 5.0, 0.5)*1e6
    annual = st.number_input("Annual consumption (m bbl)", 0.5, 20.0, 3.5, 0.1)*1e6
    maxh = st.slider("Max hedge ratio", 0.05, 1.0, 0.40, 0.05)
    otm = st.slider("OTM strike (%)", 0, 20, 5)/100
    tmin, tmax = st.slider("Tenor range (months ahead)", 1, 12, (1, 6))
    vol_mode = st.radio("Volatility", ["Flat", "Rolling realised (12m)"])
    vol = st.slider("Flat vol / fallback (%)", 15, 80, 35)/100
    floor = st.slider("Vol floor for realised (%)", 10, 50, 25)/100
    up = st.file_uploader("Optional price CSV (date, USD/bbl)", type="csv")
px = load_prices(up)
d = run(px, budget, otm, vol_mode, vol, annual, maxh, tmin, tmax, floor)
d["buy_yr"] = d.buy.dt.year; d["net"] = d.payout - d.premium

c = st.columns(5)
c[0].metric("Premium spent", f"${d.premium.sum()/1e6:.1f}m")
c[1].metric("Payout", f"${d.payout.sum()/1e6:.1f}m")
c[2].metric("Net P&L", f"${d.net.sum()/1e6:.1f}m")
c[3].metric("Barrels hedged", f"{d.bbl.sum()/1e6:.2f}m")
c[4].metric("Avg premium", f"${d.premium.sum()/d.bbl.sum():.2f}/bbl")

by = d.groupby("buy_yr").agg(premium=("premium", "sum"), payout=("payout", "sum"), net=("net", "sum"), bbl=("bbl", "sum"))
l, r = st.columns(2)
l.subheader("Net result by purchase year (USD)"); l.bar_chart(by[["premium", "payout", "net"]])
r.subheader("Spot price (USD/bbl)"); r.line_chart(px)
st.subheader("Cumulative net P&L by expiry (USD m)")
cum = d.groupby("expiry").net.sum().cumsum()/1e6; st.line_chart(cum)
st.subheader("Annual summary"); st.dataframe(by.round(0))

st.subheader("Volatility sensitivity")
rows = []
for v in (.25, .35, .45):
    x = run(px, budget, otm, "Flat", v, annual, maxh, tmin, tmax, floor)
    rows.append(dict(vol=v, premium_m=x.premium.sum()/1e6, payout_m=x.payout.sum()/1e6,
                     net_m=(x.payout.sum()-x.premium.sum())/1e6, avg_prem_bbl=x.premium.sum()/x.bbl.sum()))
st.dataframe(pd.DataFrame(rows).round(2))
with st.expander("Layer detail"):
    st.dataframe(d); st.download_button("Download CSV", d.to_csv(index=False), "layers.csv")
st.caption("Spot-only backtest: flat curve (forward = spot), Black-76 premium, settled on monthly average spot. Indicative only. Data: EIA USGC jet to Sep 2025."
