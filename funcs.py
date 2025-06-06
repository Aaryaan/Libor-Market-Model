import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m
import QuantLib as ql

def to_real_time(term, out = 'y', day_counts_in_year = 360):
    out = out.lower()
    end = term[-1].lower()
    ans = float(term[:-1])
    if end == 'y':
        if out == 'y':
            return(ans)
        elif out == 'm':
            return(12*ans)
        elif out == 'd':
            return(360*ans)
    elif end == 'm':
        if out == 'y':
            return (round(ans/12,3))
        elif out == 'm':
            return ans
        elif out == 'd':
            return ans * 30
    elif end == 'd':
        if out == 'd':
            return ans
        else:
            ans = round(ans/day_counts_in_year,3)
            if out == 'y':
                return(ans)
            elif out == 'm':
                return(ans*30)
def to_real_time_array(terms, out = 'y', t0 = '0m'):

    t0 = to_real_time(t0, out)
    time = []
    for i in terms:
        time.append(t0 + to_real_time(i, out))
    return time

def qldate_to_date(qldate):
    return(str(qldate.year()) + '-' + str(qldate.month()) + '-' + str(qldate.dayOfMonth()))

def date_diff(qldatestart, qldateend): # Years, 30 360
    time = (qldatestart.year() - qldateend.year()) + (qldatestart.month() - qldateend.month()) / 12 + ( qldatestart.dayOfMonth() - qldateend.dayOfMonth()) / 30
    return time
dateformat = '%Y-%m-%d'
def date_to_qldate(date, dateformat = dateformat):
    return ql.Date(date, dateformat)



if __name__ == '__main__':

    print(to_real_time_array(['0m','9m','30y'],'y'))



