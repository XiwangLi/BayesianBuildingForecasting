#!/usr/bin/env python

from util import *

#---------------------------------------------------------------------------
# select run name
#---------------------------------------------------------------------------

#run = '100k-2yr-var1000-min0.01-occ6to22'
#run = '100k-1yr-var1000-min0.001-occ7to20'
#run = '100k-1yr-var1000-min0.005-occ7to20'
#run = '100k-1yr-var1000-min0.01-occ7to20'
#run = '100k-1yr-var1000-min0.01-occ7to20-new-priors'
#run = '100k-1yr-var1000-min0.005-occ7to20-new-priors'
#run = '200k-1yr-var1000-min0.005-occ7to20-normal-priors'
#run = '250k-1yr-var1000-min0.005-occ7to20-normal-priors'
#run = '100k-var100-min0-subset'
#run = '100k-var100-min0.01-subset'
#run = '100k-var100-min0.001-subset'
#run = '100k-var1000-min0-rep-subset'
#run = '100k-var1000-min0.01-rep-subset'
#run = '100k-var1000-min0-rep-subset-eqpost'
#run = '100k-var1000-min0.01-rep-subset-eqpost'
#run = '100k-var1000-min0.001-rep-subset'
#run = '100k-var1000-min0.001-wider-priors'
#run = '100k-var1000-min0.001-even-wider-priors'
#run = '100k-var1000-min0.001-still-wider-priors'
#run = '100k-var1000-min0.001-fixed-seed'
#run = '100k-var5000-min0.001-fixed-seed'
#run = '100k-var10000-min0.001-fixed-seed'
#run = '100k-var25000-min0.001-perc-90-10'
#run = '100k-var25000-min0-perc-90-10'

#---------------------------------------------------------------------------
# read measured data
#---------------------------------------------------------------------------

data = pd.read_csv(os.path.join('data','bldg05.csv'), index_col='time (yyyy-mm-dd hh:mm)')
data.index = pd.to_datetime(data.index)
data = data.resample('H').mean().interpolate(method='linear')
data = data.loc[data.index.year == 2010]
num_times = len(data)

data['occ'] = (data.index.hour >= 7) & (data.index.hour < 20) & (data.index.weekday < 5)

time = np.array([dt.datetime.strptime(t,'%Y-%m-%d %H:%M') for t in data.index.strftime('%Y-%m-%d %H:%M')])
temp = np.array(data['temperature (F)'])
load = np.array(data['load (kW)'])
occ = np.array(data['occ'])
del(data)

# only use a maximum number of occupied and representative points in each temperature bin
rep = (load >= 300) & (load < 650)
bin_width = 10.0
pts_per_bin = 25
min_bin = np.floor(np.min(temp) / bin_width) * bin_width
max_bin = np.ceil(np.max(temp) / bin_width) * bin_width
temp_bins = np.arange(min_bin, max_bin+bin_width, bin_width)
obs = np.full(num_times, False)
for b in xrange(len(temp_bins)-1):
    idx = np.where(occ & rep & (temp >= temp_bins[b]) & (temp < temp_bins[b+1]))[0]
    np.random.shuffle(idx)
    idx = idx[:pts_per_bin]
    obs[idx] = True
num_obs = np.sum(obs)

#---------------------------------------------------------------------------
# scenarios and priors (values and probabilities of each parameter)
#---------------------------------------------------------------------------

num_scens = int(100e3)
num_params = 6

param_names = ['' for i in xrange(num_params)]
scens = np.full((num_scens,num_params), np.nan)

param_names[0] = 'Load Offset (kW)'
scens[:,0] = np.random.normal(loc=400, scale=25, size=num_scens)

param_names[1] = 'Heating Change Point (F)'
scens[:,1] = np.random.normal(loc=43, scale=4, size=num_scens)

param_names[2] = 'Cooling Change Point (F)'
scens[:,2] = np.random.normal(loc=60, scale=4, size=num_scens)

param_names[3] = 'Heating Slope (kW/F)'
scens[:,3] = np.random.normal(loc=-0.75, scale=2, size=num_scens)

param_names[4] = 'Deadband Slope (kW/F)'
scens[:,4] = np.random.normal(loc=0.75, scale=2, size=num_scens)

param_names[5] = 'Cooling Slope (kW/F)'
scens[:,5] = np.random.normal(loc=25, scale=3, size=num_scens)

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
    likelihood[k] = np.prod(errorPDF(load[obs], pred_load[obs]))

# P(Theta_k | X, Y)
posterior = likelihood * prior
posterior /= np.sum(posterior)

#---------------------------------------------------------------------------
# predict load
#---------------------------------------------------------------------------

pred = np.full((num_scens,num_times), np.nan)
for s in xrange(num_scens):
    cc, cp0, cp1, tc0, tc1, tc2 = scens[s,:]
    pred[s,:] = model(cc, [cp0,cp1], [tc0,tc1,tc2], temp)

pri_pred_10 = np.full(num_times, np.nan)
pri_pred_50 = np.full(num_times, np.nan)
pri_pred_90 = np.full(num_times, np.nan)
for t in xrange(num_times):
    sort_idx = np.argsort(pred[:,t])
    sum_sort_pri = np.cumsum(prior[sort_idx])
    idx_10 = np.where(sum_sort_pri >= 0.1)[0][0]
    idx_50 = np.where(sum_sort_pri >= 0.5)[0][0]
    idx_90 = np.where(sum_sort_pri >= 0.9)[0][0]
    pri_pred_10[t] = pred[:,t][sort_idx][idx_10]
    pri_pred_50[t] = pred[:,t][sort_idx][idx_50]
    pri_pred_90[t] = pred[:,t][sort_idx][idx_90]

post_pred_10 = np.full(num_times, np.nan)
post_pred_50 = np.full(num_times, np.nan)
post_pred_90 = np.full(num_times, np.nan)
for t in xrange(num_times):
    sort_idx = np.argsort(pred[:,t])
    sum_sort_post = np.cumsum(posterior[sort_idx])
    idx_10 = np.where(sum_sort_post >= 0.1)[0][0]
    idx_50 = np.where(sum_sort_post >= 0.5)[0][0]
    idx_90 = np.where(sum_sort_post >= 0.9)[0][0]
    post_pred_10[t] = pred[:,t][sort_idx][idx_10]
    post_pred_50[t] = pred[:,t][sort_idx][idx_50]
    post_pred_90[t] = pred[:,t][sort_idx][idx_90]

#---------------------------------------------------------------------------
# save variables
#---------------------------------------------------------------------------

shelf = shelve.open(run+'.out', 'n')
for var in save_load_vars:
    try:
        shelf[var] = globals()[var]
    except:
        print('unable to save %s' % var)
shelf.close()

#---------------------------------------------------------------------------

# ssh $tower 'mkdir ~/Desktop/load-models'
# scp -rq *.py data venv $tower:~/Desktop/load-models/
# ssh $tower '(cd /Users/twalter/Desktop/load-models && source venv/bin/activate && PYTHONHOME="$VIRTUAL_ENV" /opt/local/bin/python compute.py) &> /Users/twalter/Desktop/load-models/compute.log &'
# ssh $tower 'ps -A | grep compute'
# ssh $tower 'ls -Alh ~/Desktop/load-models'
# ssh $tower 'cat ~/Desktop/load-models/compute.log'
# scp $tower:~/Desktop/load-models/*.out .
# ssh $tower 'rm -r ~/Desktop/load-models'
