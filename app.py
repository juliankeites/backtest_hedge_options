import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from pandas.tseries.offsets import CustomBusinessDay

st.set_page_config(page_title="Daily fuel APO backtest", layout="wide")

def apo_price(spot, strike, times, sigma):
    # Flat forwards and a single lognormal price factor; ACT/365 fixing times.
    times=np.asarray(times,dtype=float)
    variance_ratio=np.expm1(sigma*sigma*np.minimum.outer(times,times)).mean()
    log_variance=np.log1p(variance_ratio)
    if log_variance<1e-14:
        return max(spot-strike,0.)
    z=np.sqrt(log_variance)
    d1=(np.log(spot/strike)+.5*log_variance)/z
    return max(0.,spot*norm.cdf(d1)-strike*norm.cdf(d1-z))

def load_daily(upload):
    f=pd.read_csv(upload)
    if f.shape[1]!=2: raise ValueError('CSV must have exactly two columns: date, price.')
    dates=pd.to_datetime(f.iloc[:,0],errors='raise').dt.normalize()
    p=pd.Series(pd.to_numeric(f.iloc[:,1],errors='raise').to_numpy(),index=pd.DatetimeIndex(dates)).sort_index()
    if p.index.hasnans or p.index.has_duplicates or len(p)<2:
        raise ValueError('Dates must be valid and unique; provide at least two observations.')
    if not np.isfinite(p).all() or (p<=0).any():
        raise ValueError('Prices must be finite and positive.')
    return p

def engine(px,calendar,budget,annual,ratio,otm,lo,hi,mode,flat,floor,vrp,tc):
    periods=pd.period_range(px.index.min(),px.index.max(),freq='M')
    schedules={m:pd.date_range(m.start_time,m.end_time.normalize(),freq=calendar) for m in periods}
    complete={m:dates for m,dates in schedules.items() if len(dates)>0 and dates.isin(px.index).all()}
    rv=np.log(px).diff().rolling(252,min_periods=60).std().shift(1)*np.sqrt(252)
    target=annual/12*ratio/(hi-lo+1)
    records=[]; spent={}
    for month in periods:
        dates=schedules[month]
        if len(dates)==0 or dates[0] not in px.index: continue
        buy=dates[0]; spot=float(px.loc[buy])
        realised=rv.get(buy,np.nan)
        base=flat if mode=='Flat' else max(floor,realised if pd.notna(realised) else flat)
        sigma=max(.01,base+vrp)
        candidates=[]
        for tenor in range(lo,hi+1):
            expiry_month=month+tenor
            if expiry_month not in complete: continue
            fixings=complete[expiry_month]
            times=(fixings-buy).days.to_numpy()/365.
            strike=spot*(1+otm)
            premium=apo_price(spot,strike,times,sigma)*(1+tc)
            average=float(px.loc[fixings].mean())
            candidates.append((tenor,expiry_month,fixings,strike,premium,average))
        required=sum(x[4]*target for x in candidates)
        remaining=max(0.,budget-spent.get(buy.year,0.))
        scale=min(1.,remaining/required) if required>0 else 1.
        for tenor,m,fixings,strike,premium,average in candidates:
            q=target*scale; swap=(average-spot)*target
            records.append(dict(buy=buy,buy_month=buy.to_period('M').to_timestamp(),
                settlement_month=m.to_timestamp(),settlement_date=fixings[-1],tenor=tenor,
                fixing_count=len(fixings),spot=spot,call_strike=strike,swap_strike=spot,vol=sigma,
                monthly_average=average,target_bbl=target,call_bbl=q,swap_bbl=target,
                premium_bbl=premium,required_premium=premium*target,premium_paid=premium*q,
                call_received=max(average-strike,0.)*q,swap_received=max(swap,0.),swap_paid=max(-swap,0.)))
            spent[buy.year]=spent.get(buy.year,0.)+premium*q
    monthly=pd.Series({m.to_timestamp():float(px.loc[dates].mean()) for m,dates in complete.items()},dtype=float).sort_index()
    return pd.DataFrame(records),monthly,schedules,complete

st.title('Jet fuel: daily prices and average-price options')
st.caption('Trade on the first configured business day of each purchase month. Flat forward curve equals that day’s spot. '
           'APO and swap settle against the arithmetic mean of the delivery month’s configured business-day fixings.')
with st.sidebar:
    upload=st.file_uploader('Daily CSV: date, price (ISO dates recommended)',type='csv')
    unit=st.selectbox('Uploaded price unit',['USD/bbl','USD/gal'])
    budget=float(st.number_input('Annual premium budget (USD)',min_value=100_000,value=5_000_000,step=100_000))
    annual=float(st.number_input('Annual consumption (bbl)',min_value=1_000,value=3_500_000,step=1_000))
    ratio=st.slider('Target hedge ratio',.05,1.,.40,.05)
    otm=st.slider('Call OTM (%)',0,20,5)/100
    lo,hi=st.slider('Delivery month offsets (1 = next month)',1,12,(1,6))
    mode=st.radio('Volatility',['Flat','Rolling daily realised'])
    flat=st.slider('Flat / fallback vol (%)',5,100,35)/100
    floor=st.slider('Realised vol floor (%)',5,80,25)/100
    vrp=st.slider('Implied volatility adjustment (vol pts)',-10,20,3)/100
    tc=st.slider('Transaction cost (% of premium)',0.,10.,2.,.5)/100
    holidays=st.text_area('Non-publication holidays (YYYY-MM-DD, comma separated)',help='Default calendar is Monday–Friday only. Enter all relevant assessment holidays. Missing expected fixings are not filled.')
st.info('Swaps retain target volumes regardless of OTM or call budget. Calls are reduced if the budget binds; '
        'the comparison then has unequal volumes. APO premiums use a two-moment lognormal approximation, not vanilla Black-76.')
if upload is None:
    st.info('Upload daily history to begin. The old embedded monthly data cannot supply first-day spots or daily APO fixings.')
    st.stop()
try:
    px=load_daily(upload)
    if unit=='USD/gal': px=px*42
    holiday_list=pd.to_datetime([x.strip() for x in holidays.split(',') if x.strip()],errors='raise')
    calendar=CustomBusinessDay(holidays=holiday_list)
    expected=pd.date_range(px.index.min(),px.index.max(),freq=calendar)
    ignored=px.index.difference(expected)
    px=px.loc[px.index.intersection(expected)]
    if len(px)<2: raise ValueError('Too few observations on configured business days.')
    d,monthly,schedules,complete=engine(px,calendar,budget,annual,ratio,otm,lo,hi,mode,flat,floor,vrp,tc)
except Exception as exc:
    st.error(str(exc)); st.stop()
if len(ignored): st.warning(f'{len(ignored)} weekend/holiday rows ignored under the configured publication calendar.')
missing=[]
for m,dates in schedules.items():
    absent=dates.difference(px.index)
    if len(absent): missing.append({'Month':str(m),'Missing fixing count':len(absent),'Missing dates':', '.join(absent.strftime('%Y-%m-%d'))})
if missing:
    st.warning('Incomplete months are excluded from settlement averages and physical-cost statistics. '
               'Check the holiday calendar; no missing prices are filled. Trades may originate in an incomplete month only if its first business-day spot exists.')
    with st.expander('Incomplete months'): st.dataframe(pd.DataFrame(missing))
if d.empty:
    st.warning('No eligible trades. Supply daily history covering purchase dates and complete future delivery months.'); st.stop()
d['call_net']=d.call_received-d.premium_paid
d['swap_net']=d.swap_received-d.swap_paid
# Keep every calendar month for cash timing, even if its physical average is unavailable.
index=pd.date_range(px.index.min().to_period('M').to_timestamp(),px.index.max().to_period('M').to_timestamp(),freq='MS')
cost=pd.DataFrame(index=index)
cost['unhedged']=monthly*annual/12
for col,date in [('premium_paid','buy_month'),('call_received','settlement_month'),('swap_received','settlement_month'),('swap_paid','settlement_month')]:
    cost[col]=d.groupby(date)[col].sum().reindex(index,fill_value=0)
cost['calls']=cost.unhedged+cost.premium_paid-cost.call_received
cost['swaps']=cost.unhedged-cost.swap_received+cost.swap_paid
cost['call_net_cash']=cost.call_received-cost.premium_paid
cost['swap_net_cash']=cost.swap_received-cost.swap_paid
fund=d.groupby(d.buy.dt.year).agg(required_premium=('required_premium','sum'),actual_spend=('premium_paid','sum'),target_bbl=('target_bbl','sum'),call_bbl=('call_bbl','sum'))
fund['budget']=budget
fund['extra_budget_needed']=(fund.required_premium-budget).clip(lower=0)
fund['unused_budget']=(budget-fund.actual_spend).clip(lower=0)
fund['target_barrels_funded_pct']=100*fund.call_bbl/fund.target_bbl
coverage=d.groupby('settlement_month')[['target_bbl','call_bbl','swap_bbl']].sum().reindex(index,fill_value=0)
coverage['call_ratio_pct']=100*coverage.call_bbl/(annual/12)
coverage['swap_ratio_pct']=100*coverage.swap_bbl/(annual/12)
coverage['target_ratio_pct']=100*ratio
full=np.isclose(coverage.target_bbl,annual/12*ratio)
achieved=coverage.loc[full,'call_ratio_pct'].mean()
def money(x): return f'{x/1e6:,.2f}m USD'
st.header('1. Funding and achieved hedge ratio')
a,b,c=st.columns(3)
a.metric('Target hedge ratio',f'{ratio:.1%}')
b.metric('Calls: fully layered months',f'{achieved:.1f}%' if pd.notna(achieved) else 'N/A')
c.metric('Sample premium spent',money(d.premium_paid.sum()))
if (fund.extra_budget_needed>.01).any(): st.warning('Premium budget is insufficient in one or more purchase years; see shortfalls below.')
else: st.success('All eligible target call trades are funded.')
st.dataframe(fund.round(2))
st.caption('Budgets reset each calendar year; no partial-year prorating. Purchases are funded chronologically, '
           'scaling tenor layers equally on each purchase date. Required premium covers eligible trades only; '
           'out-of-sample delivery months and incomplete settlement months are omitted. Achieved ratio excludes months without all intended layers.')
with st.expander('Monthly coverage'): st.dataframe(coverage.round(2))
st.header('2. Monthly physical fuel costs and hedge cash flow')
a,b=st.columns(2)
a.line_chart(cost[['unhedged','calls','swaps']]/1e6)
b.line_chart(cost[['call_net_cash','swap_net_cash']]/1e6)
st.caption('USD m. Physical bill assumes annual consumption/12 barrels per month at that month’s arithmetic average. '
           'Option premiums are paid on trade date. Both instruments settle on the last fixing date; no payment lag. '
           'Missing physical averages remain missing, not zero.')
st.header('3. Strategy comparison')
valid=cost[['unhedged','calls','swaps']].dropna()
p95=valid.quantile(.95)
cmp=pd.DataFrame([
 ['No hedge',0,0,0,0,p95.unhedged],
 ['APO calls (budget constrained)',d.premium_paid.sum(),d.call_received.sum(),0,d.call_net.sum(),p95.calls],
 ['Swaps (target volume)',0,d.swap_received.sum(),d.swap_paid.sum(),d.swap_net.sum(),p95.swaps]],
 columns=['Strategy','Premium paid','Positive settlements received','Negative settlements paid','Net hedge result','P95 monthly cost']).set_index('Strategy')
st.dataframe((cmp/1e6).round(3))
st.caption(f'USD m. P95 uses the same {len(valid)} complete physical-price months for all strategies; other columns are totals. '
           'Cash outflows displayed as positive amounts. Net result = receipts − payments − premiums.')
st.header('4. Cash timing')
a,b=st.columns(2)
a.bar_chart(cost[['premium_paid','call_received','swap_received','swap_paid']]/1e6)
b.line_chart(cost[['call_net_cash','swap_net_cash']].cumsum()/1e6)
st.header('5. Reconciliation')
checks={
 'Swap barrels equal target':np.allclose(d.swap_bbl,d.target_bbl),
 'Call barrels within target':bool((d.call_bbl<=d.target_bbl+1e-6).all()),
 'Annual premium budgets respected':bool((fund.actual_spend<=budget+.01).all()),
 'Monthly target within hedge cap':bool((coverage.target_bbl<=annual/12*ratio+1e-6).all()),
 'Call cash matches trade net':np.isclose(cost.call_net_cash.sum(),d.call_net.sum()),
 'Swap cash matches trade net':np.isclose(cost.swap_net_cash.sum(),d.swap_net.sum()),
 'Call receipts within target-volume swap positive receipts':bool((d.call_received<=d.swap_received+1e-6).all())}
if otm==0 and np.allclose(d.call_bbl,d.swap_bbl):
    checks['ATM equal-volume receipt equality']=np.allclose(d.call_received,d.swap_received)
st.dataframe(pd.DataFrame({'Check':list(checks),'Result':['PASS' if v else 'FAIL' for v in checks.values()]}),hide_index=True)
if not all(checks.values()): st.error('Reconciliation failed; review before use.')
st.header('Appendix')
for label,frame,filename,include_index in [('Trade layers',d,'trades.csv',False),('Monthly costs',cost,'monthly_costs.csv',True),('Annual funding',fund,'funding.csv',True),('Coverage',coverage,'coverage.csv',True)]:
    with st.expander(label):
        st.dataframe(frame)
        st.download_button('Download '+label,frame.to_csv(index=include_index),filename)
st.caption('Model: single-factor lognormal spot with flat forwards, zero rates and ACT/365 fixing times. '
           'Arithmetic APO premium uses lognormal matching of the first two moments of the future monthly average. '
           'This is approximate, not an exact APO price or a calibrated commodity forward-curve model. '
           'Daily realised volatility uses up to 252 prior log returns, minimum 60, annualised by sqrt(252); '
           'insufficient history uses the fallback. No future realised prices enter premium calculations. '
           'Flat/fallback volatility and its adjustment remain assumptions, not observed historical implied volatility. '
           'Trades use the first business-day observed spot (at that observation time, not a claimed opening execution). '
           'No collateral, swap margin, financing costs or assessment basis mismatch is modelled.')
