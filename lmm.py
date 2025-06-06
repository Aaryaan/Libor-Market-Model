import numpy as np
import cupy as cp
import scipy as sc
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import QuantLib as ql
import yieldcurve as yc
import time
import funcs as f
import swaption as swp

#plt.rcParams["figure.facecolor"] = "grey"
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

correlations = np.ones((4, 4)) * 0.75
for i in range(len(correlations)):
    correlations[i, i] = 1

default_lmm_options = {'model':{'correlations' : correlations,
                            'lmm_displacement': 0/100, # % to fraction earlier: 0.8
                            'lmm_skew' : 0.30227599,
                            'vol_alpha': 0.0081749, 'vol_beta': 0.00546891,
                            'vol_gamma' : 0.15334623, 'vol_d': 0.00587788,
                            }, # Earlier 0.01, 0.005, 1
                        'sim':{'paths':50000, 'lowMem':0}}

# '''

class lmm:

    def __init__(self, swaption_wrapper, options=None):

        if type(options) == type(None):
            options = default_lmm_options

        sec = swaption_wrapper.swaption
        swaption_type = swaption_wrapper.type
        self.swaption_wrapper = swaption_wrapper

        self.swaption_type = swaption_type
        self.options = options
        self.strike = swaption_wrapper.strike
        self.expiry = swaption_wrapper.expiry
        self.tenor = swaption_wrapper.tenor

        self.swaption = sec
        self.swap = sec.underlyingSwap()
        self.idx = self.swap.iborIndex()
        self.yieldcurve = self.idx.forwardingTermStructure()
        #self.current_date = self.yieldcurve.referenceDate()
        self.current_date = swaption_wrapper.date


        #self.start_disc_factor = self.swap.startDiscounts(0) # Leg 0 == Leg 1 discounting
        #self.end_disc_factor =self.swap.endDiscounts(0)

        # Value eg: self.swap.fixedRate() * self.swap.nominal() * self.swap.endDiscounts(0)
        # Do this for multiple years if multiple fixed coupons
        fixed_cash = 0
        for i, cf in enumerate(self.swap.leg(0)):
            fixed_cash += cf.amount()    # cf.date() not needed
        self.fixed_cash = fixed_cash / self.swap.nominal()

        self.daycount = self.swap.floatingDayCount()
        schedule = self.swap.floatingSchedule()
        self.calendar = schedule.calendar()
        self.expiries = schedule.dates()
        #self.tenortime = self.daycount.yearFraction(self.expiries[0], self.expiries[-1]) # This gives too high a value. 1.0138
        self.tenortime = (self.expiries[-1] - self.expiries[0]) / ((self.current_date)+ql.Period('1Y') - self.current_date)
        self.starts = self.expiries[:-1]
        self.expiries = self.expiries[1:]
        self.days_in_year = self.daycount.dayCount(self.current_date, self.current_date + ql.Period('1Y'))

        self.option_expiry = sec.exercise().dates()[0] # Swaption start
        # option_expiry = self.swap.startDate() # Same val
        self.swap_maturity = self.swap.floatingSchedule().dates()[-1] # Check if 2 bus day payment delay

        self.delta = 0.5 * ('6M' in self.idx.name() ) + 0.25 * ('3M' in self.idx.name() )
        self.yearcount = self.daycount.yearFraction(self.option_expiry,self.swap_maturity)

        self.n_libors = len(self.expiries)
        #self.n_libors = round(self.yearcount / self.delta ) # N of simulated libors. # CHeck what to do with rounding dates off

        self.paths = options['sim']['paths']
        self.displacement = options['model']['lmm_displacement'] # F = L + displacement. We diffuse F with LMM formula for - rates

        # Rebonato Vol params
        self.vol_alpha = options['model']['vol_alpha']
        self.vol_beta = self.options['model']['vol_beta']
        self.vol_gamma = self.options['model']['vol_gamma']
        self.vol_d = self.options['model']['vol_d']

        # Correlation loading
        self.corr = options['model']['correlations']

        # Skew Loading
        # Skew is the power of L(t,T) in vol(t,T,L) in GBM diffusion (BARUCH LMM)
        # dL => vol(t,T,L) = u * dt + vol(t,T) * L(t,T) ^ skew * dW
        # Skew = 1 => Regular GBM, Skew = 0 => Normal diff
        if 'lmm_skew' in options['model']:
            self.lmm_skew = options['model']['lmm_skew']
        else:
            self.lmm_skew = 1

        self.cudaoverride = cp.cuda.runtime.getDeviceCount() > 0
        self.lowMem = options['sim']['lowMem']
        self.diffusion = None
        self.calculatedprice = None

    def change_model_params(self, params):
        # Supply new params as [alpha, beta, gamma]
        self.vol_alpha, self.vol_beta, self.vol_gamma, self.vol_d, self.lmm_skew = params
        self.diffusion = None   # Reset Diffusions
        self.extracted_diffusions = None

    def vol(self,current_time, expiration_times):
        if cp.cuda.runtime.getDeviceCount() > 0 and self.cudaoverride != 0:
            xp = cp
        else:
            xp = np
        vol = self.vol_d + xp.exp(-self.vol_gamma * (expiration_times - current_time)) * (
                    self.vol_alpha + self.vol_beta * (expiration_times - current_time))
        # Might want to add + const term to vol as initial vol is v small
        #vol = self.vol_d + np.exp(-self.vol_gamma * (expiration_times - current_time)) * (self.vol_alpha + self.vol_beta * (expiration_times - current_time))
        return(vol)

        # Future expansion: correl term structure p = p0 exp(- eta * t ) # Check

    def expected_payoffold(self, libors ): # Does not work for fixed leg periods > 1 as assumes 1 cashflow
        # Expected forward NPV

        rates = self.delta * libors
        rates = rates.transpose()

        discounts = 1/(1+rates)
        discounts = np.cumprod(discounts, axis = 1)

        npv_float = np.sum(discounts * rates, axis=1).transpose() # Same as 1 - D(t0, tf), probably faster. Shift.
        npv_fixed = discounts[:,-1] * self.fixed_cash

        difference = npv_float.reshape(-1,1) - npv_fixed.reshape(-1,1)  # Payer swaption payoff
        difference = difference * ( 1 -  2 * int(self.swaption_wrapper.isShort)) # Reverse diff, if swap is short, payer reciever

        single_price = np.average(np.max(difference, axis = 1, initial = 0 ) )
        if self.swaption_type == 'Single':
            return single_price # Float cashflows is n x 1 vec
        elif self.swaption_type == 'Straddle':
            reversed_price = np.average(np.max(-difference, axis = 1, initial = 0 ) )
            return single_price + reversed_price

    def expected_payoff(self): # Does not work for fixed leg periods > 1 as assumes 1 cashflow
        # Expected forward NPV

        if type(self.diffusion) == type(None):
            self.diffuse()

        if cp.cuda.runtime.getDeviceCount() > 0 and self.cudaoverride != 0:
            xp = cp
            if type(self.extracted_diffusions) == type(np.array([1])):
                self.extracted_diffusions = cp.asarray(self.extracted_diffusions)
        else:
            xp = np

        rates = self.delta * self.extracted_diffusions
        rates = rates.transpose()

        discounts = 1/(1+rates)
        discounts = xp.cumprod(discounts, axis = 1)

        npv_float = xp.sum(discounts * rates, axis=1).transpose() # Same as 1 - D(t0, tf), probably faster. Shift.
        npv_fixed = discounts[:,-1] * self.fixed_cash

        difference = npv_float.reshape(-1,1) - npv_fixed.reshape(-1,1)  # Payer swaption payoff
        difference = difference * ( 1 -  2 * int(self.swaption_wrapper.isShort)) # Reverse diff, if swap is short, reciever

        if self.swaption_type == 'Single':
            difference[difference <= 0] = 0
            price = xp.average(difference)

        elif self.swaption_type == 'Straddle':
            difference = xp.abs(difference)
            price = xp.average(difference)

        if cp.cuda.runtime.getDeviceCount() > 0 and self.cudaoverride != 0:
            price = cp.asnumpy(price)

            if self.lowMem:
                del self.extracted_diffusions
            else:
                self.extracted_diffusions = cp.asnumpy(self.extracted_diffusions)
            cp._default_memory_pool.free_all_blocks()

        if self.lowMem:
            del self.diffusion
            self.diffusion = None
        return float(price)

    def diffuseold2(self, get_full_diffusion = 0):

        if type(self.diffusion) != type(None):# and ql.Settings.instance().evaluationDate == self.current_date:
            if get_full_diffusion:
                return self.diffusion
            else:
                return self.extracted_diffusions

        ql.Settings.instance().evaluationDate = self.current_date

        #timesteps = self.calendar.businessDaysBetween(self.option_expiry, self.expiries[-2]) # Only simulates option exp onwards
        timesteps = self.calendar.businessDaysBetween(self.current_date, self.expiries[-2])
        #timesteps = 200 # Change to daily later

        diffusion = np.zeros((self.n_libors, self.paths, timesteps+1))# n_libor x Path x Timestep matrix will be created

        bus_days_in_year = self.calendar.businessDaysBetween(self.current_date, self.current_date+ql.Period('2Y')) / 2
        expiration_times = []
        for expiry in self.expiries:
            #expiration_times.append(self.daycount.yearFraction(self.current_date, expiry))
            expiration_times.append(self.calendar.businessDaysBetween(self.current_date, expiry)/bus_days_in_year)
        expiration_times = np.array(expiration_times).reshape( (-1,1) ) # Column vec

        '''
        f0 = []
        for i in range(self.n_libors):
            f0.append(self.yieldcurve.forwardRate(self.starts[i], #self.expiries[i] - self.idx.tenor(),
                                                  self.expiries[i],
                                                  self.idx.dayCounter(),
                                                  ql.Simple,ql.Annual).rate() + self.displacement )
        #f0 = [f0] * self.paths
        #f0 = np.array(f0)
        #'''
        f0 = self.fwdRatesFromCurve() + self.displacement # Override fwds
        f0 = np.array(f0).reshape((-1,1))
        f0 = np.repeat(f0, self.paths, 1)
        diffusion[:,:,0] = f0

        extracted_diffusions = np.zeros((self.n_libors, self.paths))
        extracted_diffusions[0,:] = diffusion[0,:,0]

        #time = self.daycount.yearFraction(self.current_date, self.option_expiry) # Start of swap
        #time = (self.expiries[-2] - self.current_date) / ((self.current_date)+ql.Period('1Y') - self.current_date)
        #time_start = (self.option_expiry - self.current_date) / ((self.current_date)+ql.Period('1Y') - self.current_date)
        #time_start = f.date_diff( self.option_expiry, self.current_date )
        time_start = self.calendar.businessDaysBetween(self.current_date, self.option_expiry) / bus_days_in_year
        time_start = 0 # Start from today

        #time_end = (self.expiries[-2] - self.current_date) / ((self.current_date)+ql.Period('1Y') - self.current_date)
        #time_end = f.date_diff(self.expiries[-2], self.current_date )
        time_end = self.calendar.businessDaysBetween(self.current_date, self.expiries[-2]) / bus_days_in_year

        sim_date = self.option_expiry
        sim_date = self.current_date
        time = time_start
        dt = (time_end-time_start) / timesteps

        for timestep in range(timesteps):

            sim_date = self.calendar.advance(sim_date, ql.Period('1D'))
            time += dt

            # n x paths
            #dw = np.random.normal(0,1, (1, self.paths)) * np.sqrt(dt)
            #dw = np.repeat(dw, self.n_libors, 0)   # For constant correlation, single brownian LMM
            dw = np.sqrt(dt) * np.random.multivariate_normal(np.zeros((self.n_libors)), self.corr, self.paths ).transpose()

            #'''
            if self.lmm_skew != 0:
                # Drifts is (n x paths) matrix
                drifts = np.zeros((self.n_libors, self.paths))
                for i in range(self.n_libors):
                    # not using Frozen drift approx
                    corrs = self.corr[i,i+1:]

                    corr_vol_i = corrs * self.vol(time,expiration_times[i+1:]).transpose()
                    rate_return_i = self.delta * (np.abs(diffusion[i+1:,:,timestep]) ** self.lmm_skew) / (1+self.delta * diffusion[i+1:,:, timestep])

                    # drift = (1 x k ) X ( k x Paths) = (1 x Paths) - for each libor
                    drift = - corr_vol_i @ rate_return_i # Dotting corr_i * vol_i with delta L /(1+delta L) # With skew
                    drifts[i, :] = drift # Drifts is 1 x paths array?
            #'''

            #print(self.vol(time, expiration_times ))
            if self.lmm_skew == 1:
                # Lognormal
                diffusion[:,:,timestep+1] = (1 + self.vol(time, expiration_times ) * (drifts * dt + dw )) * diffusion[:,:,timestep]

            if self.lmm_skew == 0:
                #Normal Diffusion, not theoretically correct, needs drift term. Only use for reference.
                diffusion[:, :, timestep + 1] = diffusion[:,:, timestep] + self.vol(time, expiration_times) * dw

            else:
                # General skew
                diffusion[:, :, timestep + 1] = diffusion[:,:,timestep] + (self.vol(time,
                    expiration_times ) * (drifts * dt + dw ) * np.abs(diffusion[:,:,timestep])**self.lmm_skew)

            if sim_date in self.starts:
                for i, date in enumerate(self.starts):
                    if date == sim_date:
                        extracted_diffusions[i,:] = diffusion[i,:,timestep+1]
                        #diffusion[i,:,timestep+1] = 0 # To kill Expired libors
            1
            #
        extracted_diffusions -= self.displacement
        diffusion = diffusion - self.displacement
        self.diffusion = diffusion
        self.extracted_diffusions = extracted_diffusions
        if get_full_diffusion:
            return diffusion
        else:
            return extracted_diffusions

    def diffuseold(self, get_full_diffusion = 0):

        if type(self.diffusion) != type(None):# and ql.Settings.instance().evaluationDate == self.current_date:
            if get_full_diffusion:
                return self.diffusion
            else:
                return self.extracted_diffusions

        corr = self.corr
        if cp.cuda.runtime.getDeviceCount() > 0 and self.cudaoverride != 0:
            corr = cp.asarray(self.corr)
            self.useCuda(1)
            xp = cp

        else:
            self.useCuda(0)
            xp = np
        #print(xp)

        ql.Settings.instance().evaluationDate = self.current_date

        #timesteps = self.calendar.businessDaysBetween(self.option_expiry, self.expiries[-2]) # Only simulates option exp onwards
        timesteps = self.calendar.businessDaysBetween(self.current_date, self.expiries[-2])
        #timesteps = 200 # Change to daily later

        diffusion = xp.zeros((self.n_libors, self.paths, timesteps+1))# n_libor x Path x Timestep matrix will be created

        bus_days_in_year = self.calendar.businessDaysBetween(self.current_date, self.current_date+ql.Period('2Y')) / 2
        expiration_times = []
        for expiry in self.expiries:
            #expiration_times.append(self.daycount.yearFraction(self.current_date, expiry))
            expiration_times.append(self.calendar.businessDaysBetween(self.current_date, expiry)/bus_days_in_year)
        expiration_times = xp.array(expiration_times).reshape( (-1,1) ) # Column vec

        f0 = self.fwdRatesFromCurve() + self.displacement # Override fwds
        f0 = xp.array(f0).reshape((-1,1))
        f0 = xp.repeat(f0, self.paths, 1)
        diffusion[:,:,0] = f0

        extracted_diffusions = xp.zeros((self.n_libors, self.paths))
        extracted_diffusions[0,:] = diffusion[0,:,0]

        time_start = self.calendar.businessDaysBetween(self.current_date, self.option_expiry) / bus_days_in_year
        time_start = 0 # Start from today

        time_end = self.calendar.businessDaysBetween(self.current_date, self.expiries[-2]) / bus_days_in_year

        sim_date = self.option_expiry
        sim_date = self.current_date
        time = time_start
        dt = (time_end-time_start) / timesteps



        for timestep in range(timesteps):

            sim_date = self.calendar.advance(sim_date, ql.Period('1D'))
            time += dt

            # n x paths
            dw = xp.sqrt(dt) * xp.random.multivariate_normal(xp.zeros((self.n_libors)), corr, self.paths ).transpose()

            #'''
            if self.lmm_skew != 0:
                # Drifts is (n x paths) matrix
                drifts = xp.zeros((self.n_libors, self.paths))
                for i in range(self.n_libors):
                    # not using Frozen drift approx
                    corrs = corr[i,i+1:]

                    corr_vol_i = corrs * self.vol(time,expiration_times[i+1:]).transpose()
                    rate_return_i = xp.abs(diffusion[i+1:,:,timestep]) ** self.lmm_skew / (1 / self.delta + diffusion[i+1:,:, timestep])

                    # drift = (1 x k ) X ( k x Paths) = (1 x Paths) - for each libor
                    drift = - corr_vol_i @ rate_return_i # Dotting corr_i * vol_i with delta L /(1+delta L) # With skew
                    drifts[i, :] = drift # Drifts is 1 x paths array?
            #'''

            #print(self.vol(time, expiration_times ))
            if self.lmm_skew == 1:
                # Lognormal
                diffusion[:,:,timestep+1] = (1 + self.vol(time, expiration_times ) * (drifts * dt + dw )) * diffusion[:,:,timestep]

            if self.lmm_skew == 0:
                #Normal Diffusion, not theoretically correct, needs drift term. Only use for reference.
                diffusion[:, :, timestep + 1] = diffusion[:,:, timestep] + self.vol(time, expiration_times) * dw

            else:
                # General skew
                diffusion[:, :, timestep + 1] = diffusion[:,:,timestep] + (self.vol(time,
                    expiration_times ) * (drifts * dt + dw ) * xp.abs(diffusion[:,:,timestep])**self.lmm_skew)

            if sim_date in self.starts:
                for i, date in enumerate(self.starts):
                    if date == sim_date:
                        extracted_diffusions[i,:] = diffusion[i,:,timestep+1]
                        #diffusion[i,:,timestep+1] = 0 # To kill Expired libors
            1
        del corr # Loaded into GPU
        extracted_diffusions -= self.displacement
        diffusion = diffusion - self.displacement
        '''
        if xp == cp:
            self.diffusion = cp.asnumpy(diffusion)
            self.extracted_diffusions = cp.asnumpy(extracted_diffusions)
        else:
            self.diffusion = diffusion
            self.extracted_diffusions = extracted_diffusions
        #'''
        if cp.cuda.runtime.getDeviceCount() > 0 and self.cudaoverride != 0:
            if self.lowMem == 0:
                self.diffusion = cp.asnumpy(diffusion) # Eats too much mem
        if self.lowMem:
            del diffusion
        self.extracted_diffusions = extracted_diffusions
        if get_full_diffusion:
            return diffusion
        else:
            return extracted_diffusions

    def diffuse(self, get_full_diffusion = 0):

        if type(self.diffusion) != type(None):# and ql.Settings.instance().evaluationDate == self.current_date:
            if get_full_diffusion:
                return self.diffusion
            else:
                return self.extracted_diffusions

        corr = self.corr
        if cp.cuda.runtime.getDeviceCount() > 0 and self.cudaoverride != 0:
            corr = cp.asarray(self.corr)
            self.useCuda(1)
            xp = cp

        else:
            self.useCuda(0)
            xp = np
        #print(xp)

        ql.Settings.instance().evaluationDate = self.current_date

        timesteps = self.calendar.businessDaysBetween(self.current_date, self.starts[0]) # FIX TO OPTION END!

        diffusion = xp.zeros((self.n_libors, self.paths, timesteps+1))# n_libor x Path x Timestep matrix will be created

        bus_days_in_year = self.calendar.businessDaysBetween(self.current_date, self.current_date+ql.Period('2Y')) / 2
        expiration_times = []
        for expiry in self.expiries:
            #expiration_times.append(self.daycount.yearFraction(self.current_date, expiry))
            expiration_times.append(self.calendar.businessDaysBetween(self.current_date, expiry)/bus_days_in_year)
        expiration_times = xp.array(expiration_times).reshape( (-1,1) ) # Column vec

        f0 = self.fwdRatesFromCurve() + self.displacement # Override fwds
        f0 = xp.array(f0).reshape((-1,1))
        f0 = xp.repeat(f0, self.paths, 1)
        diffusion[:,:,0] = f0

        extracted_diffusions = xp.zeros((self.n_libors, self.paths))
        extracted_diffusions[0,:] = diffusion[0,:,0]

        time_start = 0 # Start from today

        # Sim till option must be exercised. Then if Market NPV Float > NPV Fixed, exercise.
        time_end = self.calendar.businessDaysBetween(self.current_date, self.option_expiry) / bus_days_in_year

        sim_date = self.current_date
        time = time_start
        dt = (time_end-time_start) / timesteps

        for timestep in range(timesteps):

            sim_date = self.calendar.advance(sim_date, ql.Period('1D'))
            time += dt

            # n x paths
            dw = xp.sqrt(dt) * xp.random.multivariate_normal(xp.zeros((self.n_libors)), corr, self.paths ).transpose()

            #'''
            if self.lmm_skew != 0:
                # Drifts is (n x paths) matrix
                drifts = xp.zeros((self.n_libors, self.paths))
                for i in range(self.n_libors):
                    # not using Frozen drift approx
                    corrs = corr[i,i+1:]

                    corr_vol_i = corrs * self.vol(time,expiration_times[i+1:]).transpose()
                    rate_return_i = xp.abs(diffusion[i+1:,:,timestep]) ** self.lmm_skew / (1 / self.delta + diffusion[i+1:,:, timestep])

                    # drift = (1 x k ) X ( k x Paths) = (1 x Paths) - for each libor
                    drift = - corr_vol_i @ rate_return_i # Dotting corr_i * vol_i with delta L /(1+delta L) # With skew
                    drifts[i, :] = drift # Drifts is 1 x paths array?
            #'''

            #print(self.vol(time, expiration_times ))
            if self.lmm_skew == 1:
                # Lognormal
                diffusion[:,:,timestep+1] = (1 + self.vol(time, expiration_times ) * (drifts * dt + dw )) * diffusion[:,:,timestep]

            if self.lmm_skew == 0:
                #Normal Diffusion, not theoretically correct, needs drift term. Only use for reference.
                diffusion[:, :, timestep + 1] = diffusion[:,:, timestep] + self.vol(time, expiration_times) * dw

            else:
                # General skew
                diffusion[:, :, timestep + 1] = diffusion[:,:,timestep] + (self.vol(time,
                    expiration_times ) * (drifts * dt + dw ) * xp.abs(diffusion[:,:,timestep])**self.lmm_skew)

            1

        del corr # Loaded into GPU, need to clear
        diffusion = diffusion - self.displacement
        self.extracted_diffusions = diffusion[:,:,-1] # Libors at option expiry

        if cp.cuda.runtime.getDeviceCount() > 0 and self.cudaoverride != 0:
            if self.lowMem == 0:
                self.diffusion = cp.asnumpy(diffusion) # Eats too much mem
        if self.lowMem:
            del diffusion

        if get_full_diffusion:
            if self.lowMem:
                print("Diffusion not stored in low mem version. Please switch to hi mem and rerun")
                return None
            return diffusion
        else:
            return extracted_diffusions

    def reset_diffusion(self):
        self.diffusion = None
        cp._default_memory_pool.free_all_blocks()

    def price(self, paramoverrides = None):
        if type(paramoverrides) != type(None):
            self.change_model_params(paramoverrides)
        #return(self.expected_payoff(self.diffuse()))
        price = self.expected_payoff()
        self.calculatedprice = price
        return(price)

    def plotLibor(self, L, save = 0):
        # '''Plot L'th libor's time paths:

        if type(self.diffusion) == type(None):
            self.price()

        M = np.max(self.diffusion[L,:,-1])
        m = np.min(self.diffusion[L,:,-1])

        color_upper = np.array([1, 0, 0])
        color_lower = np.array([0, 0, 0])
        fig, ax = plt.subplots()
        ax.set_facecolor(np.array([230, 232, 237])/255)

        if save:
            fig.set_dpi(1200)
            fig.set_size_inches(8,6, forward=True)

        plt.title(f'Diffusion of a LMM Simulated Libor (%) with Time (Days) | CEV β = {self.lmm_skew}', loc = 'center')
        plt.xlabel('Time (Days)')
        plt.ylabel('Rate (%)')
        plt.grid(True)
        plt.grid(color='white')

        plt.xlim(0,self.diffusion.shape[-1])

        # Ensure that the axis ticks only show up on the bottom and left of the plot.
        # Ticks on the right and top of the plot are generally unnecessary chartjunk.
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        # Remove bounding box
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Remove tick marks
        #plt.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on") # No work
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        for j in range(self.diffusion.shape[-2]):
            x = self.diffusion[L,j,-1]
            scale = (x-m)/(M-m)
            plt.plot(100*self.diffusion[L, j, :], color=scale*color_upper+(1-scale)*color_lower)

        if m < 0:
            # Zero line
            plt.axhline(y=0, lw=1, color='black')
        plt.axvline(x=0, lw=1, color = 'black')

        # Fair rate
        #plt.plot(100*self.swap.fairRate() * np.ones(self.diffusion.shape[-1]), '--', color='black', )

        # Initial Rate
        plt.plot(100*self.diffusion[L, 0, 0] * np.ones(self.diffusion.shape[-1]), '--', color='white', )

        if save:
            plt.savefig('./plots/simexample/beta-'+str(self.lmm_skew)+'_'+str(time.time())+'.svg', format='svg', dpi=1200, bbox_inches="tight")
        plt.show()
        #plt.savefig('destination_path.eps', format='eps', dpi=1000, bbox_inches="tight")

    def volSmile(self):

        self.paths = 10000
        f = self.fixed_cash

        strikelist = [-100, -50, -25, 0, 25, 50, 100]
        #strikelist = [0]
        vollist = []

        t = self.tenortime
        for strike in strikelist:
            if strike < 0:
                self.swaption_wrapper.isShort = 1
            else:
                self.swaption_wrapper.isShort = 0

            k = self.fixed_cash+ strike / 1e4
            self.fixed_cash = k
            price = self.price()

            #print(price)
            def optim(vol):
                #print(vol)
                d1 = (np.log(f/k) + (t*vol**2)/2)/vol/np.sqrt(t)
                d2 = d1 - vol * np.sqrt(t)
                if strike >=0 :
                    c = f * sc.stats.norm.cdf(d1) - k * sc.stats.norm.cdf(d2) # Forward prem, no discounting
                    return (price-c)**2
                elif strike < 0:
                    p = k * sc.stats.norm.cdf(-d2) - f * sc.stats.norm.cdf(-d1)
                    return (price-p)**2

            optimised = sc.optimize.minimize(optim, 0.1, method='Nelder-Mead', bounds=((0.0001,1),),
                                                tol=1e-28, options={'maxiter':1000, 'disp':True})
            vollist.append(optimised.x[0])

            self.fixed_cash = f
        return vollist

    def fwdRatesFromCurve(self):
        val = []
        for i, cf in enumerate(self.swap.leg(1)):
            val.append(cf.amount() / self.delta)
        return np.array(val)

    def distribution(self):
        # Dist of last libor
        plt.hist(self.diffusion[3,:,-2])
        plt.show()

    def monteCarloNoise(self, paths, maxiter = 50, plot = 0):
        self.paths = paths
        plist = []
        for i in range(maxiter):
            plist.append(self.price())
            self.diffusion = None
        if plot:
            plt.hist(plist)
            plt.show()
        return np.array(plist)

    def useCuda(self, choice = 1):
        self.cudaoverride = choice
        if choice == 0:
            if type(self.diffusion) != type(None):
                self.diffusion = cp.asnumpy(self.diffusion)
            cp._default_memory_pool.free_all_blocks()
        elif choice == 1:
            if type(self.diffusion) != type(None):
                self.diffusion = cp.asarray(self.diffusion)

    def averageTime(self, paths, iters=10):
        self.paths = paths
        tlist = []
        self.diffusion = None
        for i in range(iters):
            t1 = time.time()
            price=self.price()
            t2 = time.time()
            tlist.append(t2-t1)
            self.diffusion = None
        return tlist

if __name__ == '__main__':
    rundate = yc.testpd
    default_lmm_options['sim']['lowMem'] = 0
    ycurve, ychandle, ind = yc.gen_yield_curve(rundate, 1)
    print(f'Running on {rundate}')
    single = swp.create_single_swaption(yc.testqldate, 0, '1y', '1y', ychandle)

    #mod = lmm(single)
    #mod.price()


    ''' Time analysis
    single = swp.create_single_swaption(yc.testqldate, 0, '1y', '1y', ychandle)
    mod = lmm(single)
    lc = mod.averageTime(500) # Average time 0.729
    lc = mod.averageTime(1000) # Average time 0.734 s
    lc = mod.averageTime(5000) # Average time 0.744 s
    lc = mod.averageTime(10000) # Average time 0.746 secs
    lc = mod.averageTime(40000) # Average time 0.83 secs
    lc = mod.averageTime(50000) # Average time 0.941 secs
    lc = mod.averageTime(100000) # Average time 0.959 secs
    mod.useCuda(0)
    ln = mod.averageTime(500) # Average time 0.155 s
    ln = mod.averageTime(1000) # Average time 0.219 s
    ln = mod.averageTime(5000) # Average time 0.889 s
    ln = mod.averageTime(10000) # Average time 1.66 secs
    ln = mod.averageTime(40000) # Average time 7.59 secs
    ln = mod.averageTime(50000) # Average time 10.38 secs
    ln = mod.averageTime(100000) # Average time 23.33 secs
    mod.useCuda(1)
    #'''

    ''' MC Variance analysis
    single = swp.create_single_swaption(yc.testqldate, 0, '1y', '1y', ychandle)
    mod = lmm(single)
    p = mod.monteCarloNoise(500).var(ddof=1) # 1.3007545622878867e-08
    p = mod.monteCarloNoise(1000) # 6.480844471574592e-09
    p = mod.monteCarloNoise(5000) # 1.5704192680180737e-09
    p = mod.monteCarloNoise(10000) # 7.998102359045837e-10
    p = mod.monteCarloNoise(40000) # 1.9763573849473187e-10
    p = mod.monteCarloNoise(50000) # 1.6956865265571542e-10
    p = mod.monteCarloNoise(100000) # 1.1891321004994538e-10
    #'''


    ''' Asymmetric prices across opposite strikes. Displacement? Log normal?
    diff_params = [0.00005, 0.000005, 0.8, 0.0505, 0.01]
    s0 = swp.create_single_swaption(yc.testqldate, 0, '1y', '1y', ychandle)
    sp = swp.create_single_swaption(yc.testqldate, 25, '1y', '1y', ychandle)
    sn = swp.create_single_swaption(yc.testqldate, -25, '1y', '1y', ychandle)
    m0 = lmm(s0)
    m0.change_model_params(diff_params)
    p0 = m0.price()
    mp = lmm(sp)
    mp.change_model_params(diff_params)
    pp = mp.price()
    mn = lmm(sn)
    mn.change_model_params(diff_params)
    pn = mn.price()
    print(f'Neg Str:{100*pn:.4f}\nATM Str:{100*p0:.4f}\nPos Str:{100*pp:.4f}')

    #m0.plotLibor(3)
    #'''


    '''
    single = swp.create_single_swaption(yc.testqldate, 0, '1y', '1y', ychandle)
    straddle = swp.create_straddle_swaption(yc.testqldate, 1, 1, ychandle)
    m1 = lmm(single)
    m2 = lmm(straddle)
    p1 = m1.price()
    p2 = m2.price()
    print(f"Prices:\nSingle: {p1:.4f}\nStraddle: {p2:0.4f}\n")
    #'''



    '''
    t0 = time.time()
    model = lmm(wswaption, default_lmm_options)
    t1 = time.time()
    print(t1-t0)
    diffused = model.diffuse()
    t2 = time.time()
    fwd_price = model.expected_payoff(diffused)
    t3 = time.time()
    print(t2-t1) # 7.5 secs for 40k paths daily sim
    print(t3-t2)
    # model.price() # Same thing all in 1
    #'''


    ''' Plotting Sample paths 
    temp = swp.create_single_swaption(yc.testqldate, 0, '1y', '1y', ychandle)
    plotbetaequallow = lmm(temp)
    plotbetaequallow.change_model_params([0.001,0.001,1,0.001,0.11])
    plotbetaequallow.plotLibor(3,1)
    
    plotbetaequalhi = lmm(temp)
    plotbetaequalhi.change_model_params([0.1,0.1,1,0.1,0.96])
    plotbetaequalhi.plotLibor(3,1)
    #'''

    ''' Plotting vol Smile for different betas
    single = swp.create_single_swaption(yc.testqldate, 0, '1y', '1y', ychandle)
    mod = lmm(single)
    mod.change_model_params([0.1, 0.1, 1, 0.1, 0.96])
    vol = mod.volSmile()
    # [0.17916903070938994, 0.1789677048426475, 0.17424173092753706, 0.17726529218533552, 0.17872857426647643, 0.18042922617086588]

    single = swp.create_single_swaption(yc.testqldate, 0, '1y', '1y', ychandle)
    mod = lmm(single)
    mod.change_model_params([0.005,0.01,1,0.001,0.11]) # Take higher vols or 0 prices and hence noisy vol smile
    vol = mod.volSmile()
    # [0.24469230844650175, 0.21231252300583264, 0.20008583263190224, 0.17854023908064598, 0.17326524429070506, 0.1658328033404589, 0.1458119651350089]
    #'''

    print(1)