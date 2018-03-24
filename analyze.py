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
# load variables
#---------------------------------------------------------------------------

shelf = shelve.open(run+'.out')
for var in save_load_vars:
    try:
        globals()[var] = shelf[var]
    except:
        print('unable to load %s' % var)
shelf.close()

#---------------------------------------------------------------------------
# plot
#---------------------------------------------------------------------------

with PdfPages(run+'.pdf') as pdf:

    # prior and posterior distributions of parameters
    for k in xrange(num_params):
        fig, ax = plt.subplots(1)
        CDFPlot(ax, scens[:,k], prior, color='blue', label='prior')
        CDFPlot(ax, scens[:,k], posterior, color='green', label='posterior')
        plt.xlabel(param_names[k])
        plt.ylabel('Probability')
        plt.legend(loc='upper left', edgecolor='black', fancybox=False)
        pdf.savefig()

    # temperature vs. load (with priors)
    plt.figure()
    plt.title('priors')
    plt.scatter(temp[obs], load[obs], color='black', facecolor='none', label='measured')
    sort = np.argsort(temp[obs])
    for i in xrange(50):
        s = np.random.randint(num_scens)
        cc, cp0, cp1, tc0, tc1, tc2 = scens[s,:]
        pred = model(cc, [cp0,cp1], [tc0,tc1,tc2], temp)
        plt.plot(temp[obs][sort], pred[obs][sort], color='black', linewidth=1, alpha=0.5)
    plt.plot(temp[obs][sort], pri_pred_10[obs][sort], color='blue', linewidth=3, alpha=0.5, label='predicted (10th %ile)')
    plt.plot(temp[obs][sort], pri_pred_50[obs][sort], color='blue', linewidth=3, label='predicted (median)')
    plt.plot(temp[obs][sort], pri_pred_90[obs][sort], color='blue', linewidth=3, alpha=0.5, label='predicted (90th %ile)')
    plt.xlabel('Temperature (F)')
    plt.ylabel('Load (kW)')
    plt.legend(loc='upper left', edgecolor='black', fancybox=False)
    pdf.savefig()

    # temperature vs. load (with posteriors)
    plt.figure()
    plt.title('posteriors')
    plt.scatter(temp[obs], load[obs], color='black', facecolor='none', label='measured')
    sort = np.argsort(temp[obs])
    plt.plot(temp[obs][sort], post_pred_10[obs][sort], color='blue', linewidth=3, alpha=0.5, label='predicted (10th %ile)')
    plt.plot(temp[obs][sort], post_pred_50[obs][sort], color='blue', linewidth=3, label='predicted (median)')
    plt.plot(temp[obs][sort], post_pred_90[obs][sort], color='blue', linewidth=3, alpha=0.5, label='predicted (90th %ile)')
    # maximum posterior
    s = np.where(posterior == np.max(posterior))[0][0]
    cc, cp0, cp1, tc0, tc1, tc2 = scens[s,:]
    temp_rng = [np.min(temp[obs]), cp0, cp1, np.max(temp[obs])]
    pred = model(cc, [cp0,cp1], [tc0,tc1,tc2], temp_rng)
    plt.plot(temp_rng, pred, color='red', linewidth=3, label='predicted (max post)')
    # median of each parameter
    med_params = []
    for k in xrange(num_params):
        sort_idx = np.argsort(scens[:,k])
        sum_sort_post = np.cumsum(posterior[sort_idx])
        idx_50 = np.where(sum_sort_post >= 0.5)[0][0]
        med_params.append(scens[:,k][sort_idx][idx_50])
    cc, cp0, cp1, tc0, tc1, tc2 = med_params
    temp_rng = [np.min(temp[obs]), cp0, cp1, np.max(temp[obs])]
    pred = model(cc, [cp0,cp1], [tc0,tc1,tc2], temp_rng)
    plt.plot(temp_rng, pred, color='green', linewidth=3, label='predicted (med param)')
    plt.xlabel('Temperature (F)')
    plt.ylabel('Load (kW)')
    plt.legend(loc='upper left', edgecolor='black', fancybox=False)
    pdf.savefig()

    # error distribution
    plt.figure()
    e = np.linspace(-500, 500, 250)
    plt.plot(e, errorPDF(e, np.zeros(len(e))))
    plt.xlabel('Load Prediction Error (kW)')
    plt.ylabel('Probability')
    pdf.savefig()

    # measured and predicted load vs. time
    load_occ = load
    load_occ[~occ] = np.nan
    pred_10_occ = post_pred_10
    pred_10_occ[~occ] = np.nan
    pred_50_occ = post_pred_50
    pred_50_occ[~occ] = np.nan
    pred_90_occ = post_pred_90
    pred_90_occ[~occ] = np.nan
    weeks = [[dt.datetime(2010,2,8), dt.datetime(2010,2,12)],
             [dt.datetime(2010,8,16), dt.datetime(2010,8,20)]]
    for week in weeks:
        start_day, end_day = week
        idx = (time >= start_day) & (time <= end_day+dt.timedelta(days=1))
        plt.figure()
        plt.plot(time[idx], load_occ[idx], color='black', label='measured')
        plt.plot(time[idx], pred_10_occ[idx], color='blue', alpha=0.5, label='predicted (10th %ile)')
        plt.plot(time[idx], pred_50_occ[idx], color='blue', label='predicted (median)')
        plt.plot(time[idx], pred_90_occ[idx], color='blue', alpha=0.5, label='predicted (90th %ile)')
        plt.ylabel('Load (kW)')
        plt.legend(loc='upper left', edgecolor='black', fancybox=False)
        plt.xticks(np.array([start_day+dt.timedelta(days=t+0.5) for t in xrange((end_day-start_day).days+1)]))
        plt.xlim(start_day, end_day+dt.timedelta(days=1))
        plt.gca().xaxis.set_major_formatter(DateFormatter('%a\n%I%p'))
        pdf.savefig()

    # measured load vs. predicted load
    plt.figure()
    plt.scatter(load[obs], post_pred_50[obs], color='black', facecolor='none')
    min_load = min(np.min(load[obs]), np.min(post_pred_50[obs]))
    max_load = max(np.max(load[obs]), np.max(post_pred_50[obs]))
    plt.plot([min_load,max_load], [min_load,max_load], color='blue', linewidth=3)
    plt.xlim((min_load,max_load))
    plt.ylim((min_load,max_load))
    plt.xlabel('Load (kW) - measured')
    plt.ylabel('Load (kW) - predicted (median)')
    pdf.savefig()

#---------------------------------------------------------------------------
