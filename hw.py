import numpy as np
import QuantLib as ql
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import yieldcurve as yc
import swaption as swp
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

# HW options

default_hw_options = {'model':{'mr':0.09024337, 'vol':0.00802598}}

# Code

class hw:
    def __init__(self, swaption_wrapper, options = default_hw_options):

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
        self.mr = options['model']['mr']
        self.vol = options['model']['vol']
        self.df_till_expiry = self.ychandle.discount(self.date + ql.Period(self.expiry))

    def change_model_params(self, params):
        #self.mr, self.vol = params
        self.vol = params[0]

    def pricer(self):
        ql.Settings.instance().evaluationDate = self.date
        model = ql.HullWhite(self.ychandle, self.mr, self.vol)
        self.swaption.setPricingEngine(ql.JamshidianSwaptionEngine(model, self.ychandle))
        price = self.swaption.NPV() # REMOVE DISCOUNTING
        if self.type == 'Straddle':
            self.swaption2.setPricingEngine(ql.JamshidianSwaptionEngine(model, self.ychandle))
            reverse_price = self.swaption2.NPV()
            price += reverse_price
        return price / self.df_till_expiry

    def price(self, vol_params = None):
        if type(vol_params) != type(None):
            self.change_model_params(vol_params)
        return self.pricer()


def plotHW(save=0):
    #''' For simulating

    # Lemon Yellow green
    color_lower= np.array([207, 172, 51]) /255
    color_upper=np.array([85, 109, 48]) /255

    # Dark Green
    #color_upper=np.array([16, 105, 103]) /255 /1.4
    #color_lower=np.array([79, 188, 156]) /255

    ntimesteps = 425
    simtime = 2

    #hw_model = ql.HullWhite(yc.ychandle, default_hw_options['model']['mr'], default_hw_options['model']['vol'])
    hw = ql.HullWhiteProcess(yc.ychandle, default_hw_options['model']['mr'], default_hw_options['model']['vol'])
    rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(ntimesteps, ql.UniformRandomGenerator()))
    seq = ql.GaussianPathGenerator(hw, simtime, ntimesteps, rng, False)

    def generate_paths(num_paths, timestep):
        arr = np.zeros((num_paths, timestep+1))
        for i in range(num_paths):
            sample_path = seq.next()
            path = sample_path.value()
            time = [path.time(j) for j in range(len(path))]
            value = [path[j] for j in range(len(path))]
            arr[i, :] = np.array(value)
        return np.array(time), arr

    npaths = 250
    time, paths = generate_paths(npaths, ntimesteps)
    M = np.max(paths[:,-1])
    m = np.min(paths[:,-1])

    fixed = ql.HullWhiteProcess(yc.ychandle, default_hw_options['model']['mr'], 0)
    rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(ntimesteps, ql.UniformRandomGenerator()))
    seq = ql.GaussianPathGenerator(fixed, simtime, ntimesteps, rng, False)
    _, fixed = generate_paths(1, ntimesteps)

    # Plotting
    fig, ax = plt.subplots()
    ax.set_facecolor(np.array([230, 232, 237]) / 255)

    if save:
        fig.set_dpi(1200)
        fig.set_size_inches(8, 6, forward=True)

    plt.title(f'Diffusion of a Hull-White Simulated Libor (%) with Time (Years)', loc='center')
    plt.xlabel('Time (Years)')
    plt.ylabel('Rate (%)')
    plt.grid(True)
    plt.grid(color='white')

    plt.xlim(0,time[-1])

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
    # plt.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on") # No work
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    for i in range(npaths):
        scale = (paths[i,-1]-m)/(M-m)
        plt.plot(time, 100*paths[i, :], color = color_upper*scale + color_lower*(1-scale))

    if m < 0:
        # Zero line
        plt.axhline(y=0, lw=1, color='black')
    plt.axvline(x=0, lw=1, color='black')

    # Initial Rate
    #plt.axhline(y=100*paths[0,0], lw=1, color='white')
    plt.plot(time, fixed[0,:]*100, '--', color = 'white' )

    if save:
        plt.savefig('./plots/simexample/hw'+ str(time.time()) + '.svg', format='svg',
                    dpi=1200, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':

    ycurve, ychandle, ind = yc.gen_yield_curve(yc.testpd, 1)
    single = swp.create_single_swaption(yc.testqldate, 0, '1y', '1y', ychandle)
    #straddle = swp.create_straddle_swaption(yc.testqldate, '1y', '1y', ychandle)
    straddle = swp.create_straddle_swaption(yc.testqldate, 1, 1, ychandle)

    m1 = hw(single)
    m2 = hw(straddle)

    p1 = m1.price()
    p2 = m2.price()

    print(f"Prices:\nSingle: {p1:.4f}\nStraddle: {p2:0.4f}\n")

    print(1)

    print(1)