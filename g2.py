import numpy as np
import QuantLib as ql
import pandas as pd
import matplotlib.pyplot as plt
import yieldcurve as yc
import swaption as swp

# HW options

default_g2_options = {'model':{'g2_alpha':0.226556768, 'g2_sigma':0.0133739444, 'g2_beta':0.000108596579,
                               'g2_eta': 0.0043204245, 'g2_rho': -0.75}}

# Code

class g2:
    def __init__(self, swaption_wrapper, options = default_g2_options):

        self.swaption_wrapper = swaption_wrapper
        self.swaption = swaption_wrapper.swaption

        self.date = swaption_wrapper.date
        ql.Settings.instance().evaluationDate = self.date

        self.strike = swaption_wrapper.strike
        self.expiry = swaption_wrapper.expiry
        self.tenor = swaption_wrapper.tenor
        self.type = swaption_wrapper.type
        if self.type == 'Straddle':
            self.swaption2 = swaption_wrapper.swaption2

        self.swap = self.swaption.underlyingSwap()
        self.idx = self.swap.iborIndex()
        self.ychandle = self.idx.forwardingTermStructure()
        self.df_till_expiry = self.ychandle.discount(self.date+ql.Period(self.expiry))

        self.alpha = options['model']['g2_alpha']   # MR x
        self.sigma = options['model']['g2_sigma']   # Vol x
        self.beta = options['model']['g2_beta']     # MR y
        self.eta = options['model']['g2_eta']       # Vol y
        self.rho = options['model']['g2_rho']       # corr


    def change_model_params(self, params):
        self.alpha, self.sigma, self.beta, self.eta, self.rho = params

    def pricer(self):
        ql.Settings.instance().evaluationDate = self.date
        model = ql.G2(self.ychandle, self.alpha, self.sigma, self.beta, self.eta, self.rho)
        engine = ql.G2SwaptionEngine(model,8, 16) # Note Fd pricing engines are finite difference PEs
        self.swaption.setPricingEngine(engine) # Simpson rule integrator, range is range of gaussian std devs considered, interval is how many subdivisions we break the integral over for fineness
        price = self.swaption.NPV() # REMOVE DISCOUNTING
        if self.type == 'Straddle':
            self.swaption2.setPricingEngine(engine)
            reverse_price = self.swaption2.NPV()
            price += reverse_price
        return price / self.df_till_expiry

    def price(self, vol_params = None):
        if type(vol_params) != type(None):
            self.change_model_params(vol_params)
        return self.pricer()

    def plotG2(self, save=0):

        timestep = 50
        length = 2  # in years

        g2_process = ql.G2Process(self.alpha,self.sigma, self.beta, self.eta, self.rho)

        num_paths = 200
        grid = ql.TimeGrid(length, timestep)
        rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(2, ql.UniformRandomGenerator(125)))

        def generate_paths(num_paths, timestep):
            dz = ql.Array(2, 0.0)
            path = np.zeros(shape=(num_paths, timestep + 1))
            sample = ql.Array(2, 0.0)
            for i in range(num_paths):
                ir = ql.Array(2, 0.0)
                for j in range(timestep):
                    ir = g2_process.evolve(grid[j], ir, grid.dt(j), dz)
                    sample = rng.nextSequence().value()
                    dz[0] = sample[0]
                    dz[1] = sample[1]
                    path[i, j] = np.sum(ir)
            time = np.array([grid[j] for j in range(len(grid))])
            return time, path

        time, paths = generate_paths(num_paths, timestep)
        arr = np.zeros(shape=(num_paths, timestep))
        for i in range(num_paths):
            plt.plot(time, paths[i, :])
        plt.title("G2 Short Rate Simulation")
        plt.show()

        print(1)

        1==1


if __name__ == '__main__':
    date = '2020-05-13' # Neg Interest Rate
    #ycurve, ychandle, ind = yc.gen_yield_curve(yc.testpd, 1)
    ycurve, ychandle, ind = yc.gen_yield_curve(date, 1)
    single = swp.create_single_swaption(yc.testqldate, 0, '1y', '1y', ychandle)
    m = g2(single)
    m.plotG2()

    '''
    single = swp.create_single_swaption(yc.testqldate, 0, '1y', '1y', ychandle)
    #straddle = swp.create_straddle_swaption(yc.testqldate, '1y', '1y', ychandle)
    straddle = swp.create_straddle_swaption(yc.testqldate, 1, 1, ychandle)

    m1 = g2(single)
    m2 = g2(straddle)

    p1 = m1.price()
    p2 = m2.price()

    print(f"Prices:\nSingle: {p1:.4f}\nStraddle: {p2:0.4f}\n")
    #'''

    print(1)

