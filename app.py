import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="Fuel APO and swap backtest",layout="wide")
SCHEDULES={1:[1],2:[1,14],4:[1,5,10,15]}

def read_prices(upload,unit):
    f=pd.read_csv(upload)
    if f.shape[1]!=2: raise ValueError('CSV must contain exactly two columns: date, price.')
    dates=pd.to_datetime(f.iloc[:,0],errors='raise').dt.normalize()
    values=pd.to_numeric(f.iloc[:,1],errors='raise').to_numpy(dtype=float)
    p=pd.Series(values,index=pd.DatetimeIndex(dates)).sort_index()
    if p.index.hasnans or p.index.has_duplicates or len(p)<2:
        raise ValueError('Provide unique valid dates and at least two prices.')
    if not np.isfinite(p).all() or (p<=0).any():
        raise ValueError('All prices must be finite and positive.')
    return p*42 if unit=='USD/gal' else p

def apo_price(spot,strike,times,sigma):
    times=np.asarray(times,dtype=float)
    log_variance=np.log1p(np.expm1(sigma*sigma*np.minimum.outer(times,times)).mean())
    if log_variance<1e-14: return max(spot-strike,0.)
    z=np.sqrt(log_variance)
    d1=(np.log(spot/strike)+.5*log_variance)/z
    return max(0.,spot*norm.cdf(d1)-strike*norm.cdf(d1-z))

def simulate(px,budget,annual,ratio,otm,lo,hi,ntrades,mode,flat,floor,vrp,tc):
    months={m:group for m,group in px.groupby(px.index.to_period('M'))}
    rv=np.log(px).diff().rolling(252,min_periods=60).std().shift(1)*np.sqrt(252)
    target_layer=annual/12*ratio/ntrades/(hi-lo+1)
    rows=[]; spent={}; skipped=[]
    for month,group in months.items():
        for rank in SCHEDULES[ntrades]:
            if rank>len(group):
                skipped.append({'purchase_month':str(month),'requested_observation':rank,'available_observations':len(group)})
                continue
            buy=group.index[rank-1]; spot=float(group.loc[buy])
            realised=rv.loc[buy]
            base=flat if mode=='Flat' else max(floor,realised if pd.notna(realised) else flat)
            sigma=max(.01,base+vrp)
            candidates=[]
            for tenor in range(lo,hi+1):
                delivery=month+tenor
                if delivery not in months: continue
                fixing_prices=months[delivery]
                times=(fixing_prices.index-buy).days.to_numpy()/365.
                strike=spot*(1+otm)
                premium=apo_price(spot,strike,times,sigma)*(1+tc)
                average=float(fixing_prices.mean())
                candidates.append((tenor,delivery,fixing_prices,strike,premium,average))
            required=sum(x[4]*target_layer for x in candidates)
            available=max(0.,budget-spent.get(buy.year,0.))
            scale=min(1.,available/required) if required>0 else 1.
            for tenor,delivery,fixings,strike,premium,average in candidates:
                q=target_layer*scale
                swap=(average-spot)*target_layer
                rows.append(dict(buy=buy,purchase_month=month.to_timestamp(),trade_observation=rank,
                    settlement_month=delivery.to_timestamp(),settlement_date=fixings.index[-1],
                    tenor=tenor,fixing_count=len(fixings),spot=spot,call_strike=strike,swap_strike=spot,
                    vol=sigma,monthly_average=average,target_bbl=target_layer,swap_bbl=target_layer,
                    call_bbl=q,premium_bbl=premium,required_premium=premium*target_layer,
                    premium_paid=premium*q,call_received=max(average-strike,0.)*q,
                    swap_received=max(swap,0.),swap_paid=max(-swap,0.)))
                spent[buy.year]=spent.get(buy.year,0.)+premium*q
    averages=pd.Series({m.to_timestamp():float(g.mean()) for m,g in months.items()}).sort_index()
    return pd.DataFrame(rows),averages,pd.DataFrame(skipped)

def money(x): return f'{x/1e6:,.3f}m USD'

st.title('Daily fuel prices: APO and swap backtest')
st.caption('Every supplied date is an eligible observation. Each represented month is assumed complete, '
           'including the first and last sample months. No weekend/holiday calendar or missing-day rejection is applied.')
with st.sidebar:
    upload=st.file_uploader('Daily CSV: date, price',type='csv')
    unit=st.selectbox('Price unit',['USD/bbl','USD/gal'])
    budget=float(st.number_input('Annual premium budget (USD)',min_value=100_000,value=5_000_000,step=100_000))
    annual=float(st.number_input('Annual consumption (bbl)',min_value=1_000,value=1_200_000,step=1_000))
    ratio=st.slider('Target hedge ratio',.05,1.,.40,.05)
    ntrades=st.selectbox('Trades per purchase month',[1,2,4])
    st.caption('Observation numbers: 1 trade = 1; 2 trades = 1, 14; 4 trades = 1, 5, 10, 15. Dates come from your CSV.')
    otm=st.slider('APO call OTM (%)',0,20,5)/100
    lo,hi=st.slider('Delivery month offsets (1 = next month)',1,12,(1,6))
    mode=st.radio('Volatility',['Flat','Rolling daily realised'])
    flat=st.slider('Flat / fallback vol (%)',5,100,35)/100
    floor=st.slider('Realised volatility floor (%)',5,80,25)/100
    vrp=st.slider('Implied volatility adjustment (vol pts)',-10,20,3)/100
    tc=st.slider('Transaction cost (% of premium)',0.,10.,2.,.5)/100
monthly_target=annual/12*ratio
st.info(f'Target: {annual*ratio:,.0f} bbl annually; {monthly_target:,.0f} bbl per delivery month. '
        f'At {ntrades} purchase dates, each date targets {monthly_target/ntrades:,.0f} bbl across '
        f'{hi-lo+1} tenor layer(s): {monthly_target/ntrades/(hi-lo+1):,.2f} bbl per date/tenor. '
        'Swaps keep target volumes. Calls are reduced if the premium budget binds; volumes then differ.')
if upload is None:
    st.info('Upload your daily price CSV to start.'); st.stop()
try:
    px=read_prices(upload,unit)
    d,averages,skipped=simulate(px,budget,annual,ratio,otm,lo,hi,ntrades,mode,flat,floor,vrp,tc)
except Exception as exc:
    st.error(str(exc)); st.stop()
all_months=pd.period_range(px.index.min(),px.index.max(),freq='M')
missing=all_months.difference(px.index.to_period('M').unique())
if len(missing): st.warning('Entire months absent from CSV: '+', '.join(map(str,missing))+'. No settlement or physical average is invented for them.')
if not skipped.empty:
    st.warning('Some purchase months have fewer observations than a scheduled trade requires. Those trades are skipped, not shifted or reallocated.')
    st.dataframe(skipped)
if d.empty:
    st.warning('No eligible trades: supply more delivery-month history or shorten tenors.'); st.stop()
d['call_net']=d.call_received-d.premium_paid
d['swap_net']=d.swap_received-d.swap_paid
index=all_months.to_timestamp()
cost=pd.DataFrame(index=index)
cost['unhedged']=averages*annual/12
for col,date in [('premium_paid','purchase_month'),('call_received','settlement_month'),('swap_received','settlement_month'),('swap_paid','settlement_month')]:
    cost[col]=d.groupby(date)[col].sum().reindex(index,fill_value=0)
cost['calls']=cost.unhedged+cost.premium_paid-cost.call_received
cost['swaps']=cost.unhedged-cost.swap_received+cost.swap_paid
cost['call_net_cash']=cost.call_received-cost.premium_paid
cost['swap_net_cash']=cost.swap_received-cost.swap_paid
fund=d.groupby(d.buy.dt.year).agg(required_premium=('required_premium','sum'),actual_spend=('premium_paid','sum'),target_bbl=('target_bbl','sum'),call_bbl=('call_bbl','sum'))
fund['budget']=budget
fund['extra_budget_needed']=(fund.required_premium-budget).clip(lower=0)
fund['unused_budget']=(budget-fund.actual_spend).clip(lower=0)
fund['eligible_target_funded_pct']=100*fund.call_bbl/fund.target_bbl
coverage=d.groupby('settlement_month')[['target_bbl','call_bbl','swap_bbl']].sum().reindex(index,fill_value=0)
coverage['call_ratio_pct']=100*coverage.call_bbl/(annual/12)
coverage['swap_ratio_pct']=100*coverage.swap_bbl/(annual/12)
coverage['target_ratio_pct']=100*ratio
full=np.isclose(coverage.target_bbl,monthly_target)
achieved=coverage.loc[full,'call_ratio_pct'].mean()
st.header('1. Funding and actual hedge ratio')
a,b,c,e=st.columns(4)
a.metric('Target hedge ratio',f'{ratio:.1%}')
b.metric('Calls: fully layered delivery months',f'{achieved:.1f}%' if pd.notna(achieved) else 'N/A')
c.metric('Eligible target barrels funded',f'{100*d.call_bbl.sum()/d.target_bbl.sum():.1f}%')
e.metric('Sample premium spent',money(d.premium_paid.sum()))
if (fund.extra_budget_needed>.01).any(): st.warning('Premium budget cannot fund all eligible target calls in one or more purchase years.')
else: st.success('Premium budget funds all eligible target call trades.')
st.dataframe(fund.round(2))
st.caption('Budget resets by calendar purchase year, without partial-year prorating. Funding is chronological; '
           'all tenor layers at a trade date scale equally. Required premium covers eligible trades only. '
           'Trades settling beyond the sample or in wholly absent months are omitted. '
           'Fully layered coverage excludes opening ramp-up and months with unavailable purchase layers.')
with st.expander('Monthly hedge ratios and barrels'): st.dataframe(coverage.round(2))
st.header('2. Monthly physical fuel costs and hedge benefit')
a,b=st.columns(2)
a.line_chart(cost[['unhedged','calls','swaps']]/1e6)
b.line_chart(cost[['call_net_cash','swap_net_cash']]/1e6)
st.caption('USD m. Call cost = physical bill + premium − receipts. Swap cost = physical bill − receipts + payments. '
           'Physical volume is annual consumption/12; all supplied prices in the month receive equal averaging weight.')
st.header('3. Receipts, payments and net results')
valid=cost[['unhedged','calls','swaps']].dropna()
p95=valid.quantile(.95)
cmp=pd.DataFrame([
 ['No hedge',0,0,0,0,p95.unhedged],
 ['APO calls (budget constrained)',d.premium_paid.sum(),d.call_received.sum(),0,d.call_net.sum(),p95.calls],
 ['Swaps (target volume)',0,d.swap_received.sum(),d.swap_paid.sum(),d.swap_net.sum(),p95.swaps]],
 columns=['Strategy','Premium paid','Positive settlements received','Negative settlements paid','Net hedge result','P95 monthly fuel cost']).set_index('Strategy')
st.dataframe((cmp/1e6).round(3))
st.caption(f'USD m. Outflows shown as positive amounts. Net = receipts − payments − premiums. '
           f'P95 uses the same {len(valid)} represented physical-price months for each strategy; other values are sample totals. '
           'Swap fixed price = spot on each trade date. Call strike = that spot plus OTM.')
st.header('4. Cash timing')
a,b=st.columns(2)
a.bar_chart(cost[['premium_paid','call_received','swap_received','swap_paid']]/1e6)
b.line_chart(cost[['call_net_cash','swap_net_cash']].cumsum()/1e6)
st.header('5. Reconciliation')
checks={
 'Swap volume equals eligible target':np.allclose(d.swap_bbl,d.target_bbl),
 'Call volumes within target':bool((d.call_bbl<=d.target_bbl+1e-6).all()),
 'Annual premium budgets respected':bool((fund.actual_spend<=budget+.01).all()),
 'Delivery month hedge cap respected':bool((coverage.target_bbl<=monthly_target+1e-6).all()),
 'Call cash equals trade net':np.isclose(cost.call_net_cash.sum(),d.call_net.sum()),
 'Swap cash equals trade net':np.isclose(cost.swap_net_cash.sum(),d.swap_net.sum()),
 'Call receipts within target-volume positive swap settlements':bool((d.call_received<=d.swap_received+1e-6).all())}
if otm==0 and np.allclose(d.call_bbl,d.swap_bbl):
    checks['ATM equal-volume receipts match']=np.allclose(d.call_received,d.swap_received)
st.dataframe(pd.DataFrame({'Check':list(checks),'Result':['PASS' if x else 'FAIL' for x in checks.values()]}),hide_index=True)
if not all(checks.values()): st.error('Reconciliation failed. Review before use.')
st.header('Appendix')
for label,frame,filename,inc in [('Trade layers',d,'trades.csv',False),('Monthly costs',cost,'monthly_costs.csv',True),('Annual funding',fund,'funding.csv',True),('Monthly coverage',coverage,'coverage.csv',True)]:
    with st.expander(label):
        st.dataframe(frame)
        st.download_button('Download '+label,frame.to_csv(index=inc),filename)
st.caption('Pricing: arithmetic APO, two-moment lognormal approximation under a single-factor lognormal model, '
           'flat forwards equal to each trade-date spot, zero rates, ACT/365. Fixing calendar is precisely the dates '
           'supplied for each delivery month, assumed contractually known; realised future price values enter settlement only. '
           'Daily realised volatility uses up to 252 prior returns, minimum 60, annualised by sqrt(252), with fallback/floor. '
           'Premium paid on trade date; settlement on last supplied date of delivery month. No payment lag, collateral, '
           'margin cash flows or financing costs. Approximate research backtest, not executable pricing. '
           'Dates are ISO-formatted; duplicate dates rejected. No partial-month completeness checks are possible under the supplied-calendar assumption.')
