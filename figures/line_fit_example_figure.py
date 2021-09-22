import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import os
import sys
import glob
from scipy.interpolate import InterpolatedUnivariateSpline 
from scipy.optimize import curve_fit 
from scipy.optimize import fsolve       
sys.path.append(os.path.expanduser('~/work/python_tools/'))       
from read_ibis_rec import read_ibis_rec


## gaussian
def G(x,alpha):
    ## alpha is the FWHM
    return np.sqrt(np.log(2)/np.pi)/alpha*np.exp(-(x/alpha)**2*np.log(2))

## gaussian fit
def gf(x, b, x0, alpha):
    # b: bias (we don't measure continua)
    # x0: center
    # alpha: width
    return b-G(x-x0,alpha)

idat = read_ibis_rec('/Users/lucastarr/DST_DRIVE/201703_DST/21_march_2017/ibis_tmp/results_000/6563_nb734.sav') 
ix = 400
iy = 400

tfiles = glob.glob('/Users/lucastarr/DST_DRIVE/201703_DST/21_march_2017/ibis_tmp/results_000/original_datetime/s???.ScienceObservation_datetime_nb.npy')
time = np.load(tfiles[734])[13][1]
DST_date_str='%Y-%m-%dT%H:%M:%S.%f'
dt_obj = datetime.datetime.strptime(time, DST_date_str)

halphaline = idat[0].img[:,iy,ix]/idat[0].img[0,0,0] 
s = InterpolatedUnivariateSpline(idat[0].relwave, halphaline)
gopt, gcov = curve_fit(gf, idat[0].relwave[6:20],halphaline[6:20],p0 = [1.,0.0,0.5])

new_lam = np.linspace(idat[0].relwave[0],idat[0].relwave[-1],200)

fig,ax = plt.subplots(figsize=(6,6))

ax.plot(idat[0].relwave, halphaline,'+-')
ax.plot(new_lam, s(new_lam), 'g--')
ax.plot(idat[0].relwave, gf(idat[0].relwave,gopt[0],gopt[1],gopt[2])) 
i_lmin = s(gopt[1])
ax.plot(gopt[1],i_lmin,'rx')  
ax.plot(gopt[1]-1,s(gopt[1]-1),'go')  
ax.plot(gopt[1]+1,s(gopt[1]+1),'go')
i_half = (np.mean(s(gopt[1]+[-1,1]))+i_lmin)/2
width = fsolve(lambda x: s(x)-i_half,[-0.5,0.5])

ax.plot(width,i_half*np.array([1,1]),'r--*')

txt1 = ax.text(gopt[1]-0.1,i_lmin+0.03, r'$I_{min}$')

txt2 = ax.text(width[0]-0.05,i_half, r'$I_{half}$')
txt2.set_horizontalalignment('right')

txt3 = ax.text(gopt[1], i_half+0.01, r'$\delta\lambda$')

ax.set_xlabel(r'$\Delta\lambda$')
ax.set_ylabel(r'$I(\lambda)$')
ax.legend (('Data','Spline model','Gaussian fit'),loc = 'lower right')

txt4 = ax.text(0.725,0.3,dt_obj.strftime('%Y-%m-%d %H:%M:%S'))
txt5 = ax.text(0.725,0.4,'ix=iy=400')

figname = os.path.expanduser('~/work/20170321_ALMA/analysis/line_fitting/images/line_fit_example_figure.png')
fig.savefig(figname)
