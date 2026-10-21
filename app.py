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

def run_calls(px, budget, otm, vol_mode, vol, annual, maxh, tmin, tmax, floor, vrp, tcost):
    rv = np.log(px).diff().rolling(12, min_periods=6).std()*np.sqrt(12)
    tenors = list(range(tmin, tmax+1)); layer = maxh*(annual/12)/len(tenors)
    spent, out = {}, []
    for t in px.index:
        r_ = rv.get(t, np.nan)
        v = vol if vol_mode == "Flat" else (max(r_, floor) if not np.isnan(r_) else vol) + vrp
        v = max(v, 0.05)
        for h in tenors:
            e = t + pd.DateOffset(months=h)
            if e not in px.index: continue
            S = px[t]; K = S*(1+otm); p = b76(S, K, h/12, v)*(1+tcost)
            used = spent.get(t.year, 0); q = layer
            if used + p*q > budget: q = max(0, (budget-used)/p)
            spent[t.year] = used + p*q
            out.append(dict(buy=t, expiry=e, tenor=h, vol=v, spot=S, strike=K, prem_bbl=p,
                            bbl=q, premium=p*q, payout=max(px[e]-K, 0)*q))
    return pd.DataFrame(out)

def run_swaps(px, annual, maxh, tmin, tmax):
    tenors = list(range(tmin, tmax+1)); layer = maxh*(annual/12)/len(tenors)
    out = []
    for t in px.index:
        for h in tenors:
            e = t + pd.DateOffset(months=h)
            if e not in px.index: continue
            K = px[t]
            out.append(dict(buy=t, expiry=e, tenor=h, strike=K, bbl=layer,
                            settlement=max(px[e]-K, 0), payout=(px[e]-K)*layer))
    return pd.DataFrame(out)

def monthly_cost(px, d_calls, annual):
    mc = annual/12
    unhedged = (px*mc).rename("unhedged")
    if d_calls is None or d_calls.empty:
        hedged = unhedged.copy()
        prem_out = pd.Series(0.0, index=unhedged.index); pay_in = pd.Series(0.0, index=unhedged.index)
    else:
        prem_out = d_calls.groupby("buy").premium.sum().reindex(unhedged.index).fillna(0)
        pay_in = d_calls.groupby("expiry").payout.sum().reindex(unhedged.index).fillna(0)
        hedged = (unhedged + prem_out - pay_in).rename("hedged")
    return pd.concat([unhedged, hedged, prem_out.rename("premium_cash_out"), pay_in.rename("payout_cash_in")], axis=1).dropna()

st.title("Jet fuel hedge: what protection would we have bought?")
st.caption("Reframed for the Board: cost of insurance vs. cost of a spike, cash-flow smoothing, downside protection, and comparison to swaps. Spot-only backtest, flat curve, Black-76 pricing, monthly-average settlement. Indicative only.")

with st.sidebar:
    st.header("Programme")
    budget = st.number_input("Annual premium budget (USD m)", 0.5, 50.0, 5.0, 0.5)*1e6
    annual = st.number_input("Annual consumption (m bbl)", 0.5, 20.0, 3.5, 0.1)*1e6
    maxh = st.slider("Max hedge ratio", 0.05, 1.0, 0.40, 0.05)
    otm = st.slider("OTM strike (%)", 0, 20, 5)/100
    tmin, tmax = st.slider("Tenor range (months ahead)", 1, 12, (1, 6))
    st.header("Volatility & costs")
    vol_mode = st.radio("Volatility", ["Flat", "Rolling realised (12m)"])
    vol = st.slider("Flat vol / fallback (%)", 15, 80, 35)/100
    floor = st.slider("Vol floor for realised (%)", 10, 50, 25)/100
    vrp = st.slider("Implied-over-realised premium (vol pts)", -10, 20, 3)/100
    tcost = st.slider("Transaction cost (% of premium)", 0.0, 10.0, 2.0, 0.5)/100
    up = st.file_uploader("Optional price CSV (date, USD/bbl)", type="csv")

px = load_prices(up)
d = run_calls(px, budget, otm, vol_mode, vol, annual, maxh, tmin, tmax, floor, vrp, tcost)
d["buy_yr"] = d.buy.dt.year; d["net"] = d.payout - d.premium
sw = run_swaps(px, annual, maxh, tmin, tmax)
mc = annual/12
unh = (px*mc)
cost = monthly_cost(px, d, annual)
benefit = (cost.unhedged - cost.hedged).rename("hedge benefit (USD m)")  # positive = hedge saved money

st.header("1. Cost of insurance vs. cost of a spike")
worst_m = unh.idxmax()
call_pay_that_m = d[d.expiry == worst_m].payout.sum() - d[d.expiry == worst_m].premium.sum()
swap_pay_that_m = sw[sw.expiry == worst_m].payout.sum()
c = st.columns(4)
c[0].metric("Total premium (6y)", f"${d.premium.sum()/1e6:.1f}m")
c[1].metric("Worst month unhedged", f"${unh.max()/1e6:.1f}m", f"{worst_m:%b %Y}")
c[2].metric("Calls offset in that month", f"${call_pay_that_m/1e6:.1f}m")
c[3].metric("Swaps offset in that month", f"${swap_pay_that_m/1e6:.1f}m")
st.caption(f"In the worst single month ({worst_m:%b %Y}), unhedged fuel cost was ${unh.max()/1e6:.1f}m. "
           f"The call programme offset ${call_pay_that_m/1e6:.1f}m of that; a swap programme at the same volume would have offset ${swap_pay_that_m/1e6:.1f}m.")

st.header("2. Cash-flow smoothing: monthly fuel cost (cash basis)")
st.caption("Unhedged = spot x monthly consumption. Hedged = unhedged + premium cash OUT in the purchase month - payout cash IN in the expiry month. This reflects actual cash timing, not accrual. Because the hedge effect is small relative to the total fuel bill, the two lines overlap; use the benefit chart below to see the hedge effect clearly.")
l, r = st.columns(2)
l.line_chart(cost[["unhedged", "hedged"]])
r.line_chart(benefit/1e6)
st.caption("Right chart: hedge benefit per month, USD m. Positive = hedge reduced that month's cash cost; negative = premium outflow exceeded payout inflow that month.")
st.dataframe(cost[["unhedged", "hedged", "premium_cash_out", "payout_cash_in"]].describe().round(0))

st.header("3. Downside protection: 95th percentile of monthly cost")
p95u, p95h = cost.unhedged.quantile(.95), cost.hedged.quantile(.95)
c = st.columns(3)
c[0].metric("Unhedged P95 monthly cost", f"${p95u/1e6:.1f}m")
c[1].metric("Hedged (calls) P95", f"${p95h/1e6:.1f}m", f"-${(p95u-p95h)/1e6:.1f}m")
c[2].metric("Reduction", f"{(1-p95h/p95u)*100:.0f}%")
st.caption(f"The 95th percentile of monthly fuel cost falls from ${p95u/1e6:.1f}m to ${p95h/1e6:.1f}m with the call programme. "
           f"That ${((p95u-p95h)/1e6):.1f}m reduction in tail cost is what the ${d.premium.sum()/1e6:.1f}m premium bought.")

st.header("4. Comparison: calls vs. swaps vs. no hedge")
sw_net = sw.payout.sum()
sw_hedged = unh + sw.groupby("expiry").payout.sum().reindex(unh.index).fillna(0)
rows = [dict(strategy="No hedge", premium=0, payout=0, net=0, tail_p95=p95u),
        dict(strategy="Call options", premium=d.premium.sum(), payout=d.payout.sum(),
             net=d.net.sum(), tail_p95=p95h),
        dict(strategy="Swaps (same volume)", premium=0, payout=sw_net, net=sw_net,
             tail_p95=sw_hedged.quantile(.95))]
cmp = pd.DataFrame(rows)
cmp["premium_m"] = cmp.premium/1e6; cmp["payout_m"] = cmp.payout/1e6
cmp["net_m"] = cmp.net/1e6; cmp["tail_p95_m"] = cmp.tail_p95/1e6
st.dataframe(cmp[["strategy", "premium_m", "payout_m", "net_m", "tail_p95_m"]].round(2))
st.caption("Swaps lock in the forward price (here, spot) with no premium but full downside if prices fall. "
           "Calls cost premium but preserve upside. Tail P95 is the 95th percentile of monthly fuel cost.")

st.header("5. Cash timing: when cash actually moves")
cash = cost[["premium_cash_out", "payout_cash_in"]].copy()
cash["net_cash"] = cash.payout_cash_in - cash.premium_cash_out
cash["cum_net_cash"] = cash.net_cash.cumsum()
l2, r2 = st.columns(2)
l2.bar_chart(cash[["premium_cash_out", "payout_cash_in"]]/1e6)
r2.line_chart(cash.cum_net_cash/1e6)
st.caption("Left: monthly premium cash outflow (at purchase) vs payout cash inflow (at expiry), USD m. "
           "Right: cumulative net cash from the hedge programme, USD m. Negative cumulative cash means the programme has been a net cash drain to date; the trough shows the maximum cash committed before payouts caught up.")
st.dataframe(cash.round(0))

st.header("Appendix: detail")
with st.expander("Call option layers"):
    st.dataframe(d); st.download_button("Download calls CSV", d.to_csv(index=False), "calls.csv")
with st.expander("Swap layers"):
    st.dataframe(sw); st.download_button("Download swaps CSV", sw.to_csv(index=False), "swaps.csv")
st.caption("Data: EIA USGC jet to Sep 2025 unless a CSV is uploaded. Flat curve (forward = spot). "
           "Transaction costs applied to option premium. Swap strike = spot at purchase (flat curve proxy).")
