#!/usr/bin/env python

from util import *

# to do:
# [ ] handle nans differently (plots are connecting points with gaps in between)

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
# load variables
#---------------------------------------------------------------------------

load_vars = ['time',
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

for var in load_vars:
    try:
        dir_name = run+'_vars'
        file_name = os.path.join(dir_name, var+'.out')
        shelf = shelve.open(file_name)
        globals()[var] = shelf[var]
        shelf.close()
    except Exception as e:
        print('unable to load %s: %s' % (var,e))

#---------------------------------------------------------------------------
# plot
#---------------------------------------------------------------------------

with PdfPages(run+'.pdf') as pdf:

    # prior and posterior distributions of parameters
    fig, ax = plt.subplots(num_params, 1, figsize=(11,8.5))
    for k in xrange(num_params):
        ax[k].annotate(param_names[k], xycoords='axes fraction', xy=(0.01,0.8))
        hist(ax[k], scens[:,k], prior, color='blue', label='prior')
        hist(ax[k], scens[:,k], posterior, color='green', label='posterior')
    ax[0].legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5,1.5))
    pdf.savefig()

    # temperature vs. load
    plt.figure()
    plt.scatter(temp, load, color='black', facecolor='none')
    plt.plot(temp_rng, pred_mean_rng, color='blue', linewidth=3, label='predicted ($\mu$)')
    plt.plot(temp_rng, pred_mean_rng+3*pred_std_rng, color='blue', linewidth=3, alpha=0.5, label='predicted ($\mu+3\sigma$)')
    plt.plot(temp_rng, pred_mean_rng-3*pred_std_rng, color='blue', linewidth=3, alpha=0.5, label='predicted ($\mu-3\sigma$)')
    plt.xlabel('temperature')
    plt.ylabel('load')
    plt.legend(loc='upper left')
    pdf.savefig()

    # measured and predicted load vs. time
    plt.figure()
    plt.plot(time, load, color='black', label='measured')
    plt.plot(time, pred_mean, color='blue', label='predicted ($\mu$)')
    plt.plot(time, pred_mean+3*pred_std, color='blue', alpha=0.5, label='predicted ($\mu+3\sigma$)')
    plt.plot(time, pred_mean-3*pred_std, color='blue', alpha=0.5, label='predicted ($\mu-3\sigma$)')
    plt.ylabel('load')
    plt.legend(loc='upper left')
    plt.gca().xaxis.set_major_formatter(DateFormatter('%b'))
    pdf.savefig()

    # measured load vs. predicted load
    plt.figure()
    plt.scatter(load, pred_mean, color='black', facecolor='none')
    min_load = min(np.min(load), np.min(pred_mean))
    max_load = max(np.max(load), np.max(pred_mean))
    plt.plot([min_load,max_load], [min_load,max_load], color='blue', linewidth=3)
    plt.xlim((min_load,max_load))
    plt.ylim((min_load,max_load))
    plt.xlabel('load - measured')
    plt.ylabel('load - predicted ($\mu$)')
    pdf.savefig()

#---------------------------------------------------------------------------
