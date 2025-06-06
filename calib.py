import QuantLib as ql
import lmm
import hw
import g2
import yieldcurve as yc
import swaption as swp
import time
import datetime as dt
import pandas as pd
import numpy as np
import funcs
import scipy
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore", message="cupy.random.RandomState.multivariate_normal is")
warnings.filterwarnings("ignore", message="cupy.random.multivariate_normal is")

def loaddata(name):
    with open('./saved/'+name, 'rb') as f:
        x = pickle.load(f)
        return x

def model_prices(pricer_matrix,calib_params = None):

    model_prices = np.zeros(pricer_matrix.shape)

    for i in range(pricer_matrix.shape[0]):
        for j in range(pricer_matrix.shape[1]):
            for k in range(pricer_matrix.shape[2]):
                model_prices[i,j,k] = pricer_matrix[i,j,k].price(calib_params)

    return model_prices

def mse_cost_functiom(target_prices, pricer_matrix, calib_params ):

    calculated_prices = model_prices(pricer_matrix, calib_params)
    weights = calculated_prices == np.max(calculated_prices) # More weight to A
    scale = np.prod(weights.shape) - 1 # could be 1 also
    weights = weights.astype(np.float32) * scale
    weights += 1
    cost = np.average( ( target_prices - calculated_prices) ** 2 , weights=weights)

    return cost

def get_target_prices(calib_dates, calib_strikes, calib_expiries):

    calib_strikes_transform_dict = { -200:0, -100:1, -50:2, -25:3, 0:4, 25:5, 50:6, 100:7, 200:8}
    calib_expiries_transform_dict = {'3m':0, '1y':1, '5y':2, '10y':3}

    all_tickers = pd.read_excel( yc.path_to_data + 'swaption_details.xlsx', index_col = 0, sheet_name='Forward Premium')
    all_tickers = all_tickers.to_numpy()[:,:]
    tickers = np.zeros((len(calib_strikes), len(calib_expiries)), dtype = '|S19')   # '|S18' works but not for 10y

    for i,strike in enumerate(calib_strikes):
        for j, expiry in enumerate(calib_expiries):
            ticker = all_tickers[calib_expiries_transform_dict[expiry], calib_strikes_transform_dict[strike]]
            #print(strike, expiry, ticker)
            tickers[i,j] = str.encode(ticker) # Correct, checked

    target_prices = np.zeros((len(calib_dates), tickers.shape[0], tickers.shape[1]))

    for col in range(len(calib_expiries)):
        for row in range(len(calib_strikes)):
            ticker = tickers[row,col]
            df = pd.read_excel( yc.path_to_data + 'swaption_details.xlsx', index_col = 0, sheet_name=ticker.decode())
            df = df['Last Price']
            target_prices[:,row,col] = df[calib_dates].to_numpy() / 10000 # Price is in bips

    #print('Prices Loaded')
    return(target_prices)

def get_pricer_matrix(calib_dates, calib_strikes, calib_expiries, model, options = {}):
    # Returns a matrix of pricers that are later used to calculate price(vol)

    #seclist = []
    pmlist = [] # Date x Strike x Expiry - Same as target price array
    for i, current_date in enumerate(calib_dates):

        qldate = ql.Date(current_date, yc.dateformat)
        ql.Settings.instance().evaluationDate = qldate
        ycurve = yc.gen_yield_curve(current_date)
        ychandle = ql.YieldTermStructureHandle(ycurve)
        ind = ql.Euribor3M(ychandle)

        #sec_array_strike = []
        pm_array_strike = []
        for s, strike in enumerate(calib_strikes):
            #sec_array_exp = []
            pm_array_exp = []
            for e, expiry in enumerate(calib_expiries):
                # time_to_expiry = funcs.to_real_time(expiry)
                if strike == 0:
                    sec = swp.create_straddle_swaption(qldate, expiry, '1y', ychandle)
                else:
                    sec = swp.create_single_swaption(qldate, strike, expiry, '1y', ychandle)
                #sec_array_exp.append(sec)
                if model == 'lmm':
                    if options == {}:
                        pm_array_exp.append(lmm.lmm(sec))
                    else:
                        pm_array_exp.append(lmm.lmm(sec, options))
                elif model == 'g2':
                    pm_array_exp.append(g2.g2(sec))
                elif model == 'hw':
                    pm_array_exp.append(hw.hw(sec))
            #sec_array_strike.append(sec_array_exp)
            pm_array_strike.append(pm_array_exp)
        #seclist.append(sec_array_strike)
        pmlist.append(pm_array_strike)

    return np.array(pmlist)


def optimise_model_oneshot(calib_dates,calib_strikes,calib_expiries, model, full_return = 0, x0 = None, options_override= None, plot = 0):

    target_prices = get_target_prices(calib_dates, calib_strikes, calib_expiries)

    model_list = ['hw', 'g2', 'lmm']
    if type(options_override) != type(None):
        for key in options_override:
            if key == model:
                model_options = options_override[key]
        else:
            model_options = None
    pricer_matrix = get_pricer_matrix(calib_dates, calib_strikes, calib_expiries, model, model_options)


    def cost_wrapper(calib_params):
        return mse_cost_functiom(target_prices, pricer_matrix, calib_params)

    if model == 'hw':
        #bounds = ((0.00001,0.6),(0.000001,0.5))
        bounds = ((0.000001,0.3),)
        tol = 1e-14
        maxiter = 500
    elif model == 'g2':
        # Alpha, Sigma, Beta, Eta, rho
        bounds = ((0.0001,1), (0.0001,1), (0.0001,1), (0.0001,1), (-0.85, 0))   # Rho negative for vol hump
        tol = 1e-12
        maxiter = 200
    elif model == 'lmm':
        # Alpha, Beta, Gamma, D, Skew
        bounds = ((0.0001, 0.5), (0.00000001, 1), (0.0001, 4), (0.0001, 0.05), (0.01, 0.99))
        tol = 1e-10
        maxiter = 1

    if type(x0) == type(None):
        if model == 'hw':
            #x0 = [hw.default_hw_options['model']['mr'], hw.default_hw_options['model']['vol']]
            x0 = [hw.default_hw_options['model']['vol']]
        elif model == 'g2':
            x0 = [g2.default_g2_options['model']['g2_alpha'],g2.default_g2_options['model']['g2_sigma'],
                  g2.default_g2_options['model']['g2_beta'], g2.default_g2_options['model']['g2_eta'],
                  g2.default_g2_options['model']['g2_rho']]
            #x0 = [0.04,0.04,0.03,0.03,-0.5] # Eta gives problems
        elif model == 'lmm':
            temp = lmm.default_lmm_options['model']
            x0 = [temp['vol_alpha'], temp['vol_beta'], temp['vol_gamma'], temp['vol_d'], temp['lmm_skew']]
            x0 = [0.00857742, 0.00580225, 0.14930935, 0.00534377, 0.30187107] # 500 iters, good fit??


    options = {'maxiter':maxiter, 'disp':True}

    if type(options_override) != type(None):
        for key in options_override:
            if key not in model_list:
                options[key] = options_override[key]

    methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B' ]
    method = methods[0]

    if method == 'L-BFGS-B':
        options['eps'] =  0.001

    t1 = time.time()
    optimised = scipy.optimize.minimize(cost_wrapper, x0, method = method, bounds = bounds,
                                        tol=tol, options = options )

    t2 = time.time()
    time_taken = (t2-t1) / (len(calib_strikes)*len(calib_expiries)*len(calib_dates)) # Time taken per sec to match
    # Fair to use for LMM and G2 this as HW is calibrated for every K, T, t

    if not full_return:
        if plot:
            # ''' For plotting fit
            xopt = optimised.x
            print(f'New X: {xopt}\nOld: {x0}')
            est = model_prices(pricer_matrix)
            est2 = model_prices(pricer_matrix, xopt)

            plt.plot(est[0, :, 0], marker='o', label = 'mod')
            plt.plot(est2[0, :, 0], marker='o', label = 'est')
            plt.plot(target_prices[0, :, 0], marker='o', label = 'target')
            plt.legend()
            plt.show()
            # '''

        print('Optimization Finished')
        return optimised
    else:
        return (optimised, target_prices, pricer_matrix, time_taken)


def optimise_model_sequential( start_date, end_date, frequency = '1w'):

    qldate = ql.Date(start_date, yc.dateformat)
    rundate = yc.swapcal.advance(qldate, ql.Period('0d'))
    endqldate = ql.Date(end_date, yc.dateformat)
    strdate = str(rundate.to_date())

    all_expiries = ['3m', '1y', '5y', '10y']
    #all_expiries = [ '10y']
    all_strikes = [-200, -100, -50, -25, 0, 25, 50, 100, 200]
    #all_strikes = [ 0, 25 ] # CHANGE

    calib_strikes = {'hw':[0], 'g2':[-100, -25, 0, 25, 100], 'lmm':[-100, -25, 0, 25, 100]}
    #calib_strikes = {'hw':[0],'g2':[0, 25], 'lmm':[0, 25]} # Change
    #calib_expiries = {'hw':['1y'], 'lmm':['1y','5y']} # Note calib for every expiry separately
    #calib_strikes = [-100, -25, 0, 25, 100]
    #calib_expiries = ['3y', '1y','5y']
    iterlist = {'hw':200, 'g2':200, 'lmm' : 25 } # Put half for lmm as it will iterate twice

    model_list = ['hw', 'g2', 'lmm']
    #model_list = ['hw', 'lmm']
    #model_list = ['hw', 'g2']
    #model_list = ['lmm']
    #model_list = ['hw']

    data = [] # Grow data in a list of dicts, then just call df(data)

    first_row = {} # Contains gen info
    #first_row.append({'Calib Strikes': calib_strikes})
    first_row['Strikes'] = all_strikes
    first_row['Expiries'] = all_expiries
    first_row['Model List'] = model_list
    #first_row.append({'Calib Expiries': calib_expiries})

    count = 0

    data.append(first_row)
    while rundate > endqldate:
        print(f'Executing Date {strdate}')
        data_row = {'Date': strdate}
        skip = 0
        for model in model_list:

            model_dict = {}
            print(f'    Running {model}')
            for i, expiry in enumerate(all_expiries):
                # Calibrate
                expiry_dict = {}

                # Find previous optimal as starting value, should make optim faster
                if i > 0:
                    # Smaller expiry's x0
                    x0 = model_dict[all_expiries[i-1]]['fit']['opt']['x']
                elif len(data) >= 2:
                    # Previous timeperiod's x0
                    x0 = data[-1][model][all_expiries[0]]['fit']['opt']['x']
                else:
                    x0 = None

                options_override = {'disp': False, 'maxiter':iterlist[model]}
                if model == 'lmm':
                    try: # Prefitting to x0
                        model_options = lmm.default_lmm_options
                        model_options['sim']['paths'] = 50000
                        model_options['sim']['lowMem'] = 1
                        options_override['lmm'] = model_options
                        opt = optimise_model_oneshot([strdate], [0],
                                                                                 [expiry], model, full_return=0, x0=x0,
                                                                                 options_override=options_override)
                        x0 = opt.x
                        model_options['sim']['paths'] = 30000
                        options_override['lmm'] = model_options
                    except:
                        # If no data available on that day
                        skip = 1
                        break

                try:
                    opt, market, fitted, time_taken = optimise_model_oneshot([strdate], calib_strikes[model],
                                    [expiry], model, full_return = 1, x0=x0, options_override=options_override)
                except:
                    # If no data available on that day
                    skip = 1
                    break

                opt['time_taken'] = time_taken
                expiry_dict['fit'] = {}
                expiry_dict['fit']['market'] = market
                expiry_dict['fit']['estimated'] = model_prices(fitted)
                #expiry_dict['fit']['model'] = fitted # Cant pickle this
                expiry_dict['fit']['opt'] = opt

                del market, fitted, time_taken

                '''
                if 'Market' not in data_row:
                    data_row['Market'] = market
                #'''

                # Predict across all strikes and expirires
                try:
                    target_prices = get_target_prices([strdate], all_strikes, all_expiries)
                except: # If no data
                    skip = 1
                    break
                pricer_matrix = get_pricer_matrix([strdate], all_strikes, all_expiries, model)
                t0 = time.time()
                estimates = model_prices(pricer_matrix, opt.x)
                t1 = time.time()

                expiry_dict['pred'] = {}
                expiry_dict['pred']['market'] = target_prices
                expiry_dict['pred']['model'] = estimates
                expiry_dict['pred']['time_taken'] = t1-t0

                model_dict[expiry] = expiry_dict
            data_row[model]=model_dict
            if skip: # At model level
                break
        current_day = strdate
        qldate = yc.swapcal.advance(qldate, -ql.Period(frequency))
        rundate = yc.swapcal.advance(qldate, ql.Period('0d'))
        strdate = str(rundate.to_date())
        if skip:
            print(f'Error on {current_day}, moving on to:\n')
            continue
        print('------------------------\n')

        data.append(data_row)
        count +=1
        if count % 10 == 0:
            now = dt.datetime.now()
            savestring = './saved/' +'_'.join(model_list)+'_' + str(now.year) + '-' + str(now.month) + '-' + str(now.day)
            savestring += '_' + str(now.hour) + "-" + str(now.minute)
            savestring += f'_from_{start_date}_to_{current_day}.pkl'
            with open(savestring, "wb") as f:
                pickle.dump(data, f)
            print('Data Saved at:', savestring)

    now = dt.datetime.now()
    savestring = './saved/'+'_'.join(model_list)+'_'+str(now.year)+'-' + str(now.month) + '-' + str(now.day)
    savestring += '_' + str(now.hour) + "-" + str(now.minute)
    savestring += f'_from_{start_date}_to_{current_day}.pkl'
    with open(savestring, "wb") as f:
        pickle.dump(data, f)
    print('Data Saved at:', savestring)
    return 1

if __name__ == '__main__':
    calib_options_hw = {'model': 'hw'}
    calib_options_lmm = {'model': 'lmm'}

    calib_dates = ['2025-04-09']  # Create from calendar later # Ideally calibrate everyday

    # calib_strikes = [ -200, -100, -50, -25, 0, 25, 50, 100, 200]
    # calib_strikes = [ -100, -50, -25, 0, 25, 50, 100] # Too much for LMM?
    calib_strikes = [-100, -25, 0, 25, 100]  # Try this
    # calib_strikes = [ -50, 0, 50] # Too much for LMM
    # calib_strikes = [ 0]
    # calib_strikes = [  0 ]
    # calib_expiries = ['3m', '1y', '5y', '10y']
    calib_expiries = ['10y']

    model = 'lmm'
    temp1 = optimise_model_oneshot(calib_dates,calib_strikes,calib_expiries, model)

    #temp2 = optimise_model_sequential('2025-05-07', '2020-01-01', '1m')


    print(1)


