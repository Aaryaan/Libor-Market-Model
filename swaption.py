import funcs
import yieldcurve as yc
import QuantLib as ql

# '''
class swaptionWrapper():
    # Container class only
    # Do not put pricing fn here. THis fn feeds to pricers
    def __init__(self, swaption, type, swaption2=None, strike = 0, expiry = '1y', tenor = '1y', params = None ):
        self.swaption = swaption
        self.type = type
        if type == 'Straddle':
            self.swaption2 = swaption2
        self.params = params
        self.strike = strike
        self.expiry = expiry
        self.tenor = tenor
        self.isShort = strike < 0 # Reciever Swap - You receive fixed, pay float
        self.date = ql.Settings.instance().evaluationDate
# '''


# Generate Straddle Swaption: payer + receiver
def make_zero_fixed_swap(pricing_date: ql.Date,
                        expiry: str,
                        tenor: str,
                        yieldcurvehandle: ql.YieldTermStructureHandle,
                        ) -> tuple:
    """
    Builds the forward-starting swap (payer) that begins in 'expiry' months/years
    and runs 'tenor' yrs; returns the swap.
    """
    swap_start = yc.swapcal.advance(pricing_date, ql.Period(expiry), ql.ModifiedFollowing, False)
    swap_maturity = yc.swapcal.advance(swap_start, ql.Period(tenor), ql.ModifiedFollowing, False)

    fixed_sched = ql.Schedule(swap_start, swap_maturity, ql.Period(ql.Annual),      yc.swapcal,
                              ql.ModifiedFollowing, ql.ModifiedFollowing,
                              ql.DateGeneration.Forward, False)
    float_sched = ql.Schedule(swap_start, swap_maturity, yc.ind.tenor(),  yc.swapcal,
                              ql.ModifiedFollowing, ql.ModifiedFollowing,
                              ql.DateGeneration.Forward, False)

    # 10000 notional as quotes are in bips. Doesnt change strike
    swap = ql.VanillaSwap(ql.VanillaSwap.Payer, 10000,
                           fixed_sched, 0.0, yc.swapdcconv,
                           float_sched, yc.ind, 0.0, yc.ind.dayCounter())

    swap.setPricingEngine(ql.DiscountingSwapEngine(yieldcurvehandle))
    strike = swap.fairRate()
    return swap, strike

def create_straddle_swaption(pricing_date: ql.Date,
                      expiry_yrs: str,
                      tenor_yrs: str,
                      yieldcurvehandle: ql.YieldTermStructureHandle,
                     ) -> float:
    """
    Analytic Jamshidian price of an ATM straddle (payer+receiver),
    discounted to spot. Always ATM.
    """
    swap, strike = make_zero_fixed_swap(pricing_date, expiry_yrs, tenor_yrs, yieldcurvehandle)

    exercise = ql.EuropeanExercise(swap.startDate())
    ind = yc.ind
    swapcal = yc.swapcal
    swapdcconv = yc.swapdcconv
    # Payer leg --------------------------------------------------------
    payer_swap = ql.VanillaSwap(ql.VanillaSwap.Payer, 1,
                               swap.fixedSchedule(), strike, swapdcconv,
                               swap.floatingSchedule(), ind, 0.0,
                               ind.dayCounter())
    payer_swap.setPricingEngine(ql.DiscountingSwapEngine(yieldcurvehandle))
    payer = ql.Swaption(payer_swap, exercise)


    # Receiver leg -----------------------------------------------------
    recv_swap = ql.VanillaSwap(ql.VanillaSwap.Receiver, 1,
                               swap.fixedSchedule(), strike, swapdcconv,
                               swap.floatingSchedule(), ind, 0.0,
                               ind.dayCounter())
    recv_swap.setPricingEngine(ql.DiscountingSwapEngine(yieldcurvehandle))
    receiver = ql.Swaption(recv_swap, exercise)
    wrapped_swaption = swaptionWrapper(payer, 'Straddle', receiver, 0, expiry_yrs )
    #payer.setPricingEngine(ql.JamshidianSwaptionEngine(model, yieldcurvehandle))
    #receiver.setPricingEngine(ql.JamshidianSwaptionEngine(model, yieldcurvehandle))
    #price = payer.NPV() + receiver.NPV()
    return wrapped_swaption

def create_single_swaption(pricing_date: ql.Date,
                        strike_shift: float, # Relative, bips
                        expiry_yrs: str,    # Can input string eg '1y' and int eg. 1
                        tenor_yrs: str,
                        yieldcurvehandle: ql.YieldTermStructureHandle,
                        ) -> ql.Swaption:
    # Leg 0 is fixed leg.
    swap, strike = make_zero_fixed_swap(pricing_date, expiry_yrs, tenor_yrs, yieldcurvehandle)

    exercise = ql.EuropeanExercise(swap.startDate())

    # Since all non ATM swaps are OTM
    # Check whether price reduces with strike
    if strike_shift >= 0:
        type = ql.VanillaSwap.Payer # pay fixed, get float
    elif strike_shift < 0:
        type = ql.VanillaSwap.Receiver
    # Payment leg --------------------------------------------------------
    swap = ql.VanillaSwap(type, 1,
                                swap.fixedSchedule(), strike + strike_shift/10000, yc.swapdcconv,
                                swap.floatingSchedule(), yc.ind, 0.0,
                                yc.ind.dayCounter())
    swap.setPricingEngine(ql.DiscountingSwapEngine(yieldcurvehandle))
    swaption = ql.Swaption(swap, exercise)
    wrapped_swaption = swaptionWrapper(swaption, 'Single', None, strike_shift, expiry_yrs)
    return wrapped_swaption


if __name__ == '__main__':
    import yieldcurve as yc
    #'''
    pd = '2025-04-14'
    qldate = ql.Date(pd, yc.dateformat)
    ycurve = yc.gen_yield_curve(pd)
    mr_test = 0.03
    vol_test = 0.01
    swap, strike = make_zero_fixed_swap(qldate, '1y','10y', yc.ychandle)
    #straddle = make_straddle_swaption(qldate, '1y','10y', yc.ychandle, ql.HullWhite(yc.ychandle, mr_test, vol_test) ) # Older implem w inbuilt pricer
    straddle = create_straddle_swaption(qldate, '1y','10y', yc.ychandle )
    single = create_single_swaption(qldate, 0, '1y','1y', yc.ychandle )
    singlehigh = create_single_swaption(qldate, 50, '1y','1y', yc.ychandle )
    singlelow = create_single_swaption(qldate, -50, '1y','1y', yc.ychandle )
    #'''
    print(1)


