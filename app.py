import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="Fuel Hedge What-If", layout="wide")
RAW = {
    2020:[1.770,1.509,0.953,0.606,0.686,0.983,1.084,1.112,1.006,1.047,1.133,1.324],
    2021:[1.420,1.601,1.661,1.669,1.752,1.857,1.887,1.817,2.003,2.296,2.193,2.095],
    2022:[2.446,2.682,3.495,3.914,3.902,4.120,3.478,3.339,3.262,3.715,3.160,2.897],
    2023:[3.542,2.790,2.679,2.374,2.173,2.250,2.488,2.989,3.120,2.881,2.734,2.387],
    2024:[2.585,2.683,2.593,2.605,2.380,2.394,2.419,2.176,1.982,2.068,2.065,2.097],
    2025:[2.347,2.264,2.069,1.976,1.928,2.094,2.244,2.024,2.089],
}

@st.cache_data
def default_prices():
    return pd.Series({pd.Timestamp(y, m+1, 1): v*42
                      for y, values in RAW.items() for m, v in enumerate(values)}).sort_index()

def load_prices(upload):
    if upload is None:
        return default_prices()
    frame = pd.read_csv(upload)
    if frame.shape[1] != 2:
        raise ValueError("CSV must have exactly two columns: date, USD/bbl.")
    dates = pd.to_datetime(frame.iloc[:, 0], errors="raise").dt.to_period("M").dt.to_timestamp()
    values = pd.to_numeric(frame.iloc[:, 1], errors="raise")
    prices = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates)).sort_index()
    if prices.index.has_duplicates:
        raise ValueError("CSV has multiple observations for a month. Supply one monthly price per month.")
    if len(prices) < 2 or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Supply at least two months of finite, positive prices.")
    expected = pd.date_range(prices.index.min(), prices.index.max(), freq="MS")
    if not prices.index.equals(expected):
        raise ValueError("CSV has missing months. Supply a continuous monthly series.")
    return prices

def b76(forward, strike, years, sigma):
    scale = sigma * np.sqrt(years)
    if scale <= 0:
        return max(forward-strike, 0.0)
    d1 = (np.log(forward/strike) + 0.5*sigma*sigma*years)/scale
    return forward*norm.cdf(d1) - strike*norm.cdf(d1-scale)

def run_programme(px, budget, otm, vol_mode, vol, annual, maxh,
                  tmin, tmax, floor, vrp, tcost):
    # Current monthly average is observed at purchase; settlement is a later monthly average.
    rv = np.log(px).diff().rolling(12, min_periods=6).std()*np.sqrt(12)
    tenors = list(range(tmin, tmax+1))
    target_layer = maxh*(annual/12)/len(tenors)
    spent, records = {}, []
    for buy in px.index:
        realised = rv.loc[buy]
        base_vol = vol if vol_mode == "Flat" else max(
            realised if pd.notna(realised) else vol, floor)
        sigma = max(base_vol + vrp, 0.05)
        candidates = []
        for tenor in tenors:
            expiry = buy + pd.DateOffset(months=tenor)
            if expiry not in px.index:
                continue
            forward = float(px.loc[buy])
            strike = forward*(1+otm)
            premium_bbl = b76(forward, strike, tenor/12, sigma)*(1+tcost)
            candidates.append((tenor, expiry, forward, strike, premium_bbl))
        # Scale all tenors equally if the remaining annual budget is insufficient.
        target_premium = sum(item[4]*target_layer for item in candidates)
        remaining = max(0.0, budget-spent.get(buy.year, 0.0))
        scale = min(1.0, remaining/target_premium) if target_premium > 0 else 1.0
        for tenor, expiry, forward, strike, premium_bbl in candidates:
            volume = target_layer*scale
            settlement = float(px.loc[expiry])
            premium = premium_bbl*volume
            call_received = max(settlement-strike, 0.0)*volume
            swap_signed = (settlement-forward)*volume
            records.append(dict(
                buy=buy, expiry=expiry, tenor=tenor, vol=sigma,
                purchase_price=forward, settlement_price=settlement,
                call_strike=strike, swap_strike=forward,
                call_bbl=volume, swap_bbl=volume,
                premium_bbl=premium_bbl, premium=premium,
                call_received=call_received, call_net=call_received-premium,
                swap_received=max(swap_signed, 0.0),
                swap_paid=max(-swap_signed, 0.0), swap_net=swap_signed))
            spent[buy.year] = spent.get(buy.year, 0.0)+premium
    return pd.DataFrame(records)

def monthly_costs(px, trades, annual):
    costs = pd.DataFrame({"unhedged": px*(annual/12)})
    for column, date, source in [
        ("premium_paid", "buy", "premium"),
        ("call_received", "expiry", "call_received"),
        ("swap_received", "expiry", "swap_received"),
        ("swap_paid", "expiry", "swap_paid"),
    ]:
        costs[column] = trades.groupby(date)[source].sum().reindex(px.index, fill_value=0.0)
    costs["calls"] = costs.unhedged + costs.premium_paid - costs.call_received
    costs["swaps"] = costs.unhedged - costs.swap_received + costs.swap_paid
    costs["call_net_cash"] = costs.call_received-costs.premium_paid
    costs["swap_net_cash"] = costs.swap_received-costs.swap_paid
    return costs

def usd_m(value):
    return f"{value/1e6:,.2f}m USD"

st.title("Jet fuel hedge: what protection would we have bought?")
st.caption("Historical scenario backtest: monthly prices, flat forward curve, zero discount rate, "
           "Black-76 proxy for call premiums. Calls and swaps use identical trade dates, tenors and "
           "barrel volumes. Indicative only; not executable historical pricing.")
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

try:
    px = load_prices(up)
except (ValueError, TypeError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.stop()
if len(px) <= tmin:
    st.warning("Not enough monthly prices for the selected minimum tenor.")
    st.stop()
trades = run_programme(px, budget, otm, vol_mode, vol, annual, maxh,
                      tmin, tmax, floor, vrp, tcost)
if trades.empty:
    st.warning("No eligible trades. Adjust tenor or upload more history.")
    st.stop()
cost = monthly_costs(px, trades, annual)
st.caption(f"Sample: {px.index.min():%b %Y} to {px.index.max():%b %Y}; {len(px)} months. "
           "Premium budgets reset each calendar year and are not prorated for partial years. "
           "Trades maturing outside the sample are omitted. Opening hedge coverage is incomplete.")

st.header("1. Cost of insurance vs. cost of a spike")
worst = cost.unhedged.idxmax()
cols = st.columns(4)
cols[0].metric("Total premium (sample)", usd_m(trades.premium.sum()))
cols[1].metric("Worst month unhedged", usd_m(cost.loc[worst, "unhedged"]), f"{worst:%b %Y}")
cols[2].metric("Calls: cash-cost reduction that month", usd_m(cost.loc[worst, "call_net_cash"]))
cols[3].metric("Swaps: cash-cost reduction that month", usd_m(cost.loc[worst, "swap_net_cash"]))
st.caption("Worst-month call benefit deducts premiums actually paid in that calendar month, "
           "not premiums historically paid for the contracts expiring that month.")

st.header("2. Cash-flow smoothing: monthly fuel cost")
left, right = st.columns(2)
left.line_chart(cost[["unhedged", "calls", "swaps"]]/1e6)
right.line_chart(cost[["call_net_cash", "swap_net_cash"]]/1e6)
st.caption("USD m. Call cost = physical fuel cost + premiums paid − payouts received. "
           "Swap cost = physical fuel cost − positive settlements + negative settlements paid. "
           "Positive hedge net cash reduces the fuel bill.")
st.dataframe(cost[["unhedged", "calls", "swaps"]].describe().round(0))

st.header("3. Downside protection: 95th percentile of monthly cost")
p95 = cost[["unhedged", "calls", "swaps"]].quantile(0.95)
cols = st.columns(3)
for col, key, label in zip(cols, ["unhedged", "calls", "swaps"],
                           ["Unhedged P95", "Calls P95", "Swaps P95"]):
    col.metric(label, usd_m(p95[key]))
st.caption(f"Historical P95 cost reduction: calls {(1-p95['calls']/p95['unhedged'])*100:.1f}%; "
           f"swaps {(1-p95['swaps']/p95['unhedged'])*100:.1f}%. "
           "These are sample cash-cost quantiles, not forecasts or guaranteed protection.")

st.header("4. Comparison: calls vs. swaps vs. no hedge")
comparison = pd.DataFrame([
    {"Strategy":"No hedge", "Premium paid":0.0, "Positive settlements received":0.0,
     "Negative settlements paid":0.0, "Net hedge result":0.0,
     "P95 monthly fuel cost":p95["unhedged"]},
    {"Strategy":"Call options", "Premium paid":trades.premium.sum(),
     "Positive settlements received":trades.call_received.sum(), "Negative settlements paid":0.0,
     "Net hedge result":trades.call_net.sum(), "P95 monthly fuel cost":p95["calls"]},
    {"Strategy":"Swaps (same barrels)", "Premium paid":0.0,
     "Positive settlements received":trades.swap_received.sum(),
     "Negative settlements paid":trades.swap_paid.sum(),
     "Net hedge result":trades.swap_net.sum(), "P95 monthly fuel cost":p95["swaps"]},
]).set_index("Strategy")
st.dataframe((comparison/1e6).round(3))
st.caption("All figures USD m. Cash-out columns are positive amounts. "
           "Net hedge result = positive settlements − negative settlements − premium paid. "
           "P95 is a monthly statistic; other columns are totals over the sample.")
st.info("Swap fixed price = purchase-month price, used as a flat-forward proxy. "
        "Call strike = that price plus the selected OTM percentage. "
        "At 0% OTM, call gross payouts equal positive swap settlements trade by trade. "
        "At positive OTM, calls have a higher strike, so those receipts need not match. "
        "Swap volumes follow the actual premium-budget-constrained call volumes, not the maximum target volumes.")

st.header("5. Cash timing: when cash actually moves")
left, right = st.columns(2)
left.bar_chart(cost[["premium_paid", "call_received", "swap_received", "swap_paid"]]/1e6)
right.line_chart(cost[["call_net_cash", "swap_net_cash"]].cumsum()/1e6)
st.caption("Left: receipts and payments shown as positive amounts, USD m. "
           "Right: cumulative net hedge cash, USD m. No swap collateral, margin cash flows "
           "or financing costs are modelled.")
st.dataframe(cost.round(2))

st.header("6. Reconciliation checks")
checks = {
    "Identical call and swap barrel volumes": np.allclose(trades.call_bbl, trades.swap_bbl),
    "Swap net = receipts minus payments": np.allclose(
        trades.swap_net, trades.swap_received-trades.swap_paid),
    "Call net = payouts minus premium": np.allclose(
        trades.call_net, trades.call_received-trades.premium),
    "Call cash benefit reconciles to trade net": np.isclose(
        cost.call_net_cash.sum(), trades.call_net.sum()),
    "Swap cash benefit reconciles to trade net": np.isclose(
        cost.swap_net_cash.sum(), trades.swap_net.sum()),
    "Annual premium budgets respected": bool((
        trades.groupby(trades.buy.dt.year).premium.sum() <= budget+0.01).all()),
    "Monthly hedge volumes respect consumption cap": bool((
        trades.groupby("expiry").call_bbl.sum() <= annual/12*maxh+1e-6).all()),
}
if otm == 0:
    checks["ATM call payouts equal positive swap settlements"] = np.allclose(
        trades.call_received, trades.swap_received)
    checks["ATM call-minus-swap net equals swap payments minus premiums"] = np.isclose(
        trades.call_net.sum()-trades.swap_net.sum(),
        trades.swap_paid.sum()-trades.premium.sum())
else:
    checks["OTM call payouts do not exceed positive swap settlements"] = bool((
        trades.call_received <= trades.swap_received+1e-6).all())
st.dataframe(pd.DataFrame({"Check": list(checks),
                           "Result": ["PASS" if v else "FAIL" for v in checks.values()]}),
             hide_index=True)
if not all(checks.values()):
    st.error("Reconciliation failed. Review calculations before using results.")

st.header("Appendix: detail")
annual_premium = trades.groupby(trades.buy.dt.year).premium.sum().rename("Premium paid")
coverage = trades.groupby("expiry")[["call_bbl", "swap_bbl"]].sum().reindex(px.index, fill_value=0)
coverage["Hedge ratio"] = coverage.call_bbl/(annual/12)
with st.expander("Annual premium usage"):
    st.dataframe(annual_premium.to_frame())
with st.expander("Monthly matched volumes and hedge ratio"):
    st.dataframe(coverage)
with st.expander("Matched call and swap trade layers"):
    st.dataframe(trades)
    st.download_button("Download matched trades CSV", trades.to_csv(index=False), "matched_trades.csv")
with st.expander("Monthly fuel costs and hedge cash flows"):
    st.download_button("Download monthly costs CSV", cost.to_csv(), "monthly_costs.csv")
st.caption("Embedded price data are reproduced from the supplied app and have not been independently "
           "verified. USD/gal inputs are converted using 42 gal/bbl. Uploaded CSV prices must be USD/bbl. "
           "Monthly averages stand in for observable purchase and settlement prices; Black-76 is used "
           "as a simplified option-pricing proxy, not an exact average-price option valuation. "
           "The volatility premium is added in both flat and rolling modes. Changing this assumption "
           "can materially affect premiums, hedge volumes and backtest results.")
