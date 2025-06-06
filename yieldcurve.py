import numpy as np
import pandas as pd
import QuantLib as ql
from matplotlib import pyplot as plt
import funcs

def test_yield_curve_w_market_fra(curve,rates,dates,ind, dcconv):

    real_years = [dcconv.yearFraction(dates[0], d) for d in dates]
    curve.enableExtrapolation() # if you want to check 3m forward with actual market rates

    fwds_cc = [curve.forwardRate(d, ind.maturityDate(d), dcconv, ql.Simple,ql.Annual).rate() for d in dates] # For 3 month forward
    fwds_cc = np.array(fwds_cc)
    # fwds are perhaps slightly higher due to day counting differences
    plt.plot(real_years, 100*rates)
    plt.plot(real_years, 100*fwds_cc)

    plt.show()

    err = (fwds_cc[0] / rates[0])- 1
    print(err)
    print(1)

term = '3m'

# OPTIONS

cal = ql.TARGET() # Euribor calendar. Wont be used as Swap calendar matters more for trades.
swapcal = ql.Germany(ql.Germany.Eurex)
swapdcconv = ql.Thirty360(ql.Thirty360.European)
#DC Conv for EURIBOR6m is Actual360
#dcconv = ql.Thirty360(ql.Thirty360.ISDA) # Not for euribor
dcconv = ql.Actual360() # Checked
ind = ql.Euribor(ql.Period(term))
#ind = ql.Euribor3M() #is the same
dateformat = '%Y-%m-%d'

#----------------

# OPTIONS

path_to_data = 'F:/Projects/LSE/dissertation/code/data/Final data/'


#---------------

testpd = '2025-04-09' # Not used, but can be used as reference for testing
testqldate = ql.Date(testpd, dateformat)
#pricing_date = '2025-04-22'
#pricing_date = '2021-11-04'

# RATE LOADING FROM EXCEL-----------------------------------------------------------

forward_times = []

fra_terms = ['1x4', '2x5', '3x6', '4x7', '5x8', '6x9', '7x10', '8x11', '9x12', '10x13', '11x14', ] # Max available
fra_terms = ['1x4', '2x5', '3x6', '4x7']
#fra_terms = [] # Use to disable FRAs
fra_time = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m', '10m', '11m']
forward_times += fra_time
fra_terms_times_dict = dict(zip(fra_terms, fra_time))
fra_times_terms_dict = dict(zip(fra_time, fra_terms))

swap_terms = [1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 30, 40, 50] # Max swap terms
#swap_terms = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#swap_terms = [] # Use to disable swaps
# time = ['2y','3y','4y','5y','6y','7y','8y','9y','10y','11y','12y','15y','20y','25y','30y','40y','50y']
swap_time = []
for i in swap_terms:
    swap_time.append(str(i) + 'y')

forward_times += swap_time
swap_terms_times_dict = dict(zip(swap_terms, swap_time))
swap_times_terms_dict = dict(zip(swap_time, swap_terms))
yieldcurve = None
ychandle = None

def load_data(path_to_data):

    p = pd.read_excel( path_to_data + 'euribor3m.xlsx', index_col = 0, sheet_name='euribor3m')
    p = p.rename(columns = {'Last Price':'Spot'})
    #p = ts['EUR006M'] # DF to SERIES
    #p.loc['2025-04-15':'2025-04-22']

    # Combining FRAS

    for fra in fra_terms:
        temp = pd.read_excel(path_to_data+'euribor3m.xlsx' , sheet_name=fra)
        temp = temp.rename(columns = {'Last Price':fra})
        p = p.merge(temp, how = 'inner', on = 'Date')

    # Combining SWAPS

    for swap in swap_time:
        temp = pd.read_excel(path_to_data+'euribor3m.xlsx', sheet_name= swap )
        temp = temp.rename(columns={'Last Price': "s" + swap  })
        p = p.merge(temp, how='inner', on='Date')

    return p

def load_date( date, dataframe ):
    # real_forward_times = [0] + funcs.to_real_time_array(forward_times, 'y', t0 = '0d')
    rates = dataframe.loc[dataframe['Date'] == date].values[0, 1:]
    rates /= 100

    return rates

# YIELD CURVE GENERATION -------------------------------------------------------------------
def gen_yield_curve_from_rates(rates, pricing_date, dcconv, full_return = 0):

    global yieldcurve
    global ychandle
    global ind

    qldate = ql.Date(pricing_date, dateformat)
    ql.Settings.instance().evaluationDate = qldate

    helpers = [ql.DepositRateHelper(rates[0], ind)]

    rates_cp = np.array(rates)
    rates_cp = rates_cp[1:]

    # FRA helper
    # FRAs are ACT/360
    for i,fra in enumerate(fra_terms):
        time = fra_terms_times_dict[fra]
        #print('FRA', funcs.to_real_time(time,'m',))
        helpers.append(ql.FraRateHelper(rates_cp[i], round(funcs.to_real_time(time, 'm')), ind))

    rates_cp = rates_cp[i+1:]

    for i,swap in enumerate(swap_terms):
        time = swap_terms_times_dict[swap]
        #print('SWAP', ql.Period(time))
        # Syntax ql.SwapRateHelper(rate, tenor, calendar, fixedFrequency, fixedConvention, fixedDayCount, iborIndex)
        # FIXING CALENDAR IS SAME AS INDEX IS AN ASSUMPTION! RECHECK
        # DAY COUNTER IS CORRECT, Not sure if ISDA/european THO, recheck on that
        #'''
        helpers.append(ql.SwapRateHelper(rates_cp[i], ql.Period(time),
                                         ind.fixingCalendar(), ql.Annual, ql.ModifiedFollowing,
                                         swapdcconv, ind))
        #'''

    # EUREX CALENDAR GIVES ERROR????

    yieldcurve = ql.PiecewiseLogCubicDiscount(qldate, helpers, dcconv)
    ychandle = ql.YieldTermStructureHandle(yieldcurve)
    ind = ql.Euribor3M(ychandle)
    #print(curve.nodes())

    if full_return:
        return (yieldcurve, ychandle, ind)
    else:
        return yieldcurve

def gen_yield_curve(pricing_date, full_return=0):
    dataframe = load_data(path_to_data)
    rates = load_date(pricing_date, dataframe)

    return gen_yield_curve_from_rates(rates, pricing_date, dcconv, full_return)

def gen_ql_dates(pricing_date):

    qldate = ql.Date(pricing_date, dateformat)
    ql_dates = [qldate]
    for fra_term in fra_terms:
        ql_dates.append(qldate + ql.Period(fra_terms_times_dict[fra_term]))

    for swap_term in swap_terms:
        ql_dates.append(qldate + ql.Period(swap_terms_times_dict[swap_term]))
    return ql_dates




if __name__ == '__main__':
    pricing_date = '2025-04-22'
    #pricing_date = '2021-11-04'

    dataframe = load_data(path_to_data)
    rates = load_date(pricing_date, dataframe)
    curve = gen_yield_curve_from_rates(rates, pricing_date, dcconv)

    ql_dates = gen_ql_dates(pricing_date)

    test_yield_curve_w_market_fra(curve, rates, ql_dates, ind, dcconv)

    #plt.plot(p.loc[1,:].to_numpy()[1:])
    '''
    fig = plt.figure()
    ax = fig.gca()
    ax.grid(True)
    plt.plot( real_forward_times, p.loc[p['Date'] == date].to_numpy()[0,1:], marker = 'o')
    #plt.show()
    #'''


    '''Print zero rates
    t = np.linspace(0,10,100)
    # A zero rate, also known as a spot rate, is the interest rate on a hypothetical zero-coupon bond that matures at a specific time
    zero_rates = [curve.zeroRate(i, ql.Compounded, ql.Semiannual) for i in t]
    
    rates_ns = [100*curve.zeroRate(n, ql.Semiannual).rate() for n in t]
    
    plt.plot(t,rates_ns)
    plt.ylim(1.5,3)
    plt.show()
    print(1)
    #'''


    ''' Print equivalent rates at time t. this is not the yield curve. This is total rate till date for date
    spots = []
    tenors = []
    for d in curve.dates():
        yrs = dcconv.yearFraction(qldate, d)
        compounding = ql.Compounded
        freq = ql.Semiannual
        zero_rate = curve.zeroRate(yrs, compounding, freq)
        tenors.append(yrs)
        eq_rate = zero_rate.equivalentRate(dcconv,
                                           compounding,
                                           freq,
                                           qldate,
                                           d).rate()
        spots.append(100*eq_rate)
    plt.plot(tenors,spots)
    #'''

    ''' Instantaneous Forward rates
    curve_dates = curve.dates()
    fwds_cc = [curve.forwardRate(d, d, dcconv, ql.Simple, ql.Annual).rate() for d in
               curve_dates]  # Instantaneous Fwd, noisy as differential
    real_years = [dcconv.yearFraction(qldate, d) for d in curve_dates]
    '''