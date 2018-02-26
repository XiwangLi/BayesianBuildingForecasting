#!/usr/bin/env python

from util import *

# to do:
# [x] make priors a little wider for parameters at edge
# [x] smooth out edges of priors
# [ ] for each temp bin, only use random subset of data of max size (to avoid too much data in the middle)
# [ ] find different data with heating and cooling changepoints

#---------------------------------------------------------------------------
# select run name
#---------------------------------------------------------------------------

#run = '100k-2yr-sig1000-min0.01-occ6to22'
#run = '100k-1yr-sig1000-min0.001-occ7to20'
#run = '100k-1yr-sig1000-min0.005-occ7to20'
#run = '100k-1yr-sig1000-min0.01-occ7to20'
#run = '100k-1yr-sig1000-min0.01-occ7to20-new-priors'
#run = '100k-1yr-sig1000-min0.005-occ7to20-new-priors'
#run = '250k-1yr-sig1000-min0.005-occ7to20-normal-priors'

#---------------------------------------------------------------------------
# read measured data
#---------------------------------------------------------------------------

conv = {'time_yyyymmdd_hhmm': lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M')}
data = np.genfromtxt('data/bldg05.csv', dtype=None, delimiter=',', converters=conv, names=True)
time = data['time_yyyymmdd_hhmm']
temp = data['temperature_F']
load = data['load_kW']
yr_idx = np.array([t.year==2010 for t in time])
hr_idx = np.array([t.minute==0 for t in time])
nan_idx = (time==False) | np.isnan(temp) | np.isnan(load)
unocc_idx = np.array([t.hour < 7 for t in time]) | np.array([t.hour >= 20 for t in time]) | np.array([t.weekday() >= 5 for t in time])
idx = yr_idx & hr_idx & ~(nan_idx | unocc_idx)
time = time[idx]
temp = temp[idx]
load = load[idx]
num_obs = len(load)

#---------------------------------------------------------------------------
# scenarios and priors (values and probabilities of each parameter)
#---------------------------------------------------------------------------

num_scens = int(200e3)
num_params = 6

param_names = ['' for i in xrange(num_params)]
scens = np.full((num_scens,num_params), np.nan)

param_names[0] = 'const coeff'
scens[:,0] = np.random.normal(loc=415, scale=10, size=num_scens)

param_names[1] = 'change point 0'
scens[:,1] = np.random.normal(loc=43, scale=1.5, size=num_scens)

param_names[2] = 'change point 1'
scens[:,2] = np.random.normal(loc=59, scale=0.75, size=num_scens)

param_names[3] = 'temp coeff 0'
scens[:,3] = np.random.normal(loc=-0.75, scale=0.4, size=num_scens)

param_names[4] = 'temp coeff 1'
scens[:,4] = np.random.normal(loc=0.75, scale=0.4, size=num_scens)

param_names[5] = 'temp coeff 2'
scens[:,5] = np.random.normal(loc=25, scale=1, size=num_scens)

# P(Theta_k)
prior = np.full(num_scens, Decimal('2')/num_scens, dtype=Decimal)

#---------------------------------------------------------------------------
# compute posterior
#---------------------------------------------------------------------------

# P(X, Y | Theta_k)
likelihood = np.full(num_scens, Decimal('NaN'), dtype=Decimal)
for k in xrange(num_scens):
    cc, cp0, cp1, tc0, tc1, tc2 = scens[k,:]
    pred_load = model(cc, [cp0,cp1], [tc0,tc1,tc2], temp)
    likelihood[k] = np.prod(errorPDF(load, pred_load))

# P(Theta_k | X, Y)
posterior = likelihood * prior
posterior /= np.sum(posterior)

#---------------------------------------------------------------------------
# predict load
#---------------------------------------------------------------------------

# at measured temperatures
pred = np.full((num_scens,num_obs), np.nan)
pred_mean = np.full(num_obs, np.nan)
pred_std = np.full(num_obs, np.nan)
for s in xrange(num_scens):
    cc, cp0, cp1, tc0, tc1, tc2 = scens[s,:]
    pred[s,:] = model(cc, [cp0,cp1], [tc0,tc1,tc2], temp)
for t in xrange(num_obs):
    pred_t = np.array([Decimal(p) for p in pred[:,t]])
    pred_mean_t = np.sum(pred_t * posterior)
    pred_std_t = np.sqrt(np.sum((pred_t - pred_mean_t)**2 * posterior))
    pred_mean[t] = pred_mean_t
    pred_std[t] = pred_std_t

# across range of temperatures
temp_rng = np.linspace(np.min(temp), np.max(temp), 200)
pred_rng = np.full((num_scens,len(temp_rng)), np.nan)
pred_mean_rng = np.full(len(temp_rng), np.nan)
pred_std_rng = np.full(len(temp_rng), np.nan)
for s in xrange(num_scens):
    cc, cp0, cp1, tc0, tc1, tc2 = scens[s,:]
    pred_rng[s,:] = model(cc, [cp0,cp1], [tc0,tc1,tc2], temp_rng)
for t in xrange(len(temp_rng)):
    pred_t = np.array([Decimal(p) for p in pred_rng[:,t]])
    pred_mean_rng_t = np.sum(pred_t * posterior)
    pred_std_rng_t = np.sqrt(np.sum((pred_t - pred_mean_rng_t)**2 * posterior))
    pred_mean_rng[t] = pred_mean_rng_t
    pred_std_rng[t] = pred_std_rng_t

#---------------------------------------------------------------------------
# save variables
#---------------------------------------------------------------------------

save_vars = ['time',
             'temp',
             'load',
             'num_obs',
             'num_scens',
             'num_params',
             'param_names',
             'scens',
             'prior',
             'likelihood',
             'posterior',
             'pred',
             'pred_mean',
             'pred_std',
             'temp_rng',
             'pred_rng',
             'pred_mean_rng',
             'pred_std_rng']

for var in save_vars:
    try:
        dir_name = run+'_vars'
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        file_name = os.path.join(dir_name, var+'.out')
        shelf = shelve.open(file_name, 'n')
        shelf[var] = globals()[var]
        shelf.close()
    except Exception as e:
        print('unable to save %s: %s' % (var,e))


