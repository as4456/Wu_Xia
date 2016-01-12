import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import numpy.matlib
from TAU_P import TAU_P
from EFK_SRTSM import EFK_SRTSM
from matplotlib.backends.backend_pdf import PdfPages


current_dir = 'current directory is ' + os.getcwd()
print current_dir

maturities = []
temp = pd.read_csv('parameters/maturities.csv', header = -1, names = ['var1'])
maturities = np.array(temp['var1'])
print(maturities)
del temp

forwardrates = []
temp = pd.read_csv('results/data_forwardrates_rate.csv', header = -1)
forwardrates = np.array(temp)
del temp

forwardrates_time = []
temp = pd.read_csv('results/data_forwardrates_time.csv', header = -1)
forwardrates_time = np.array(temp)
del temp

id_2009Jan = np.where(forwardrates_time == 200901)
rlb = 0.25
fig = plt.figure(1, figsize=(14, 6))
plt.axis([id_2009Jan[0]-1, forwardrates.shape[0], 0, 10])
for column in forwardrates.T:
    plt.plot(column, ls = '--', lw= 2)
plt.legend(['ZLB','3m','6m','1y','2y','5y','7y','10y'], loc = 9, ncol= 7)  
plt.xticks(range(0, forwardrates.shape[0]+1, 24), ['1990', '1992', '1994', '1996', '1998', '2000', '2002', '2004', '2006', '2008', '2010', '2012', '2014'])
fig.savefig('plots/forward_rates.pdf',dpi=300)
plt.show()
del fig
# Read parameters file
parameters = []
temp = pd.read_csv('parameters/parameters.csv', header = -1)
parameters = np.array(temp) 
startv = parameters
epsilon = 1e3 

def decorated_cost(parameters):
    from EFK_SRTSM import EFK_SRTSM
    parameters = np.append(parameters, 0.25)
    EFK_SRTSM(parameters, maturities, forwardrates, 0)
    
xopt = fmin(decorated_cost, startv, maxiter= 1e6, full_output=True)
parameters = xopt[0]
nparam = np.append(parameters, 0.25)

print "======== shadow rate term structure model========\n"
print "The log likelihood value is: -", EFK_SRTSM(nparam, maturities, forwardrates, 0)

rhoP = parameters[0:9];
rhoP = np.reshape(rhoP, [3,3], order='F')
muP = parameters[9:12]
rhoQ1 = parameters[12]; rhoQ2 = parameters[13];
rhoQ = np.zeros((3, 3)); np.fill_diagonal(rhoQ, [rhoQ1, rhoQ2, rhoQ2])
rhoQ[1, 2] = 1
sigma = [[abs(parameters[14]), 0, 0],
        [parameters[15], abs(parameters[17]), 0],
        [parameters[16], parameters[18], abs(parameters[19])]]
delta0 = parameters[20]
sqrt_omega = parameters[21]

# compare observed and fitted forward rates(left panel of Figure 3)
forwardrates_plot = []
temp = pd.read_csv('results/forwardrates_plot.csv', header = -1)
forwardrates_plot = np.array(temp)
del temp

temp = np.array(range(1,121))
Fit_SRTSM = EFK_SRTSM(nparam,temp,forwardrates_plot,3);
del temp
id1 = np.where(forwardrates_time == 201201)
id2 = np.where(forwardrates_time == 201212)
fig = plt.figure(2)
ylb = 0; yub = 4; 
temp= np.mean(Fit_SRTSM[:, id1[0]:id2[0]+1], axis = 1)
plt.plot(range(1, 121), temp,'b-', lw = 4)
plt.plot(maturities, np.mean(forwardrates[id1[0]:id2[0]+1,:], axis = 0), 'ro', ms= 10)
plt.ylim(ylb,yub); plt.xlim(-1/4, maturities[-1]);
plt.legend(['fitted','observed'], loc = 4)
plt.title('SRTSM', fontsize=20)
plt.xticks( np.array([1/4, 1/2, 1, 2, 5, 7, 10])*12, ['','','1','2','5','7','10'])
plt.show()
fig.savefig('plots/obs_fitted_forward_rates.pdf',dpi=300)
del fig
del temp

# compute the shadow rate and compare it with the effective fed funds rate (Figure 4)
EFFR = []
temp = pd.read_csv('data/EFFR.csv', header = -1)
EFFR = np.array(temp)
del temp
SR = EFK_SRTSM(nparam,maturities,forwardrates,1)
fig = plt.figure(3)
plt.xticks(range(0, SR.shape[0]+1, 24), ['1990', '1992', '1994', '1996', '1998', '2000', '2002', '2004', '2006', '2008', '2010', '2012', '2014'])
p1 = plt.axvspan(id_2009Jan[0]-1, SR.shape[0], color='gray', alpha=0.5)
p2, = plt.plot(SR,'b',lw = 4,ms=10)
p3, = plt.plot(EFFR[360:-1],'g--',lw = 4,ms = 10)
p4, = plt.plot((0, SR.shape[0]), (rlb,rlb), 'k--', lw = 3)
plt.legend([p1, p2,p3,p4], ["ZLB period", "shadow rate", "effective federal funds rate","r"], fontsize = 12)
plt.show()
fig.savefig('plots/shadow_rate.pdf',dpi=300)
del fig
np.savetxt('shadowrate.csv', np.round(SR, decimals = 4), delimiter = ",")


# compute and plot the policy rate 
policyrate = np.append(EFFR[0:588],SR[228:])
temp = numpy.matlib.repmat(range(1960, 2014), 12, 1)*100 + np.transpose(numpy.matlib.repmat(range(1, 13), 54, 1))
time_extend = np.reshape(temp, temp.shape[0]*temp.shape[1], order = 'F')
del temp
policyrate = np.column_stack((time_extend, policyrate))

PR_plot = np.floor(np.min(policyrate[:,1])) -1
T_plot = np.max(np.floor(policyrate[:,0]/100))
fig = plt.figure(4)
plt.xticks(range(0, 648, 120), ['1960', '1970', '1980', '1990', '2000', '2010'], fontsize = 14)
p1 = plt.plot(policyrate[:,1], lw = 4, label = r'$Wu-Xia policy rate: s_t^o$')
plt.plot((0, len(time_extend)), (0.25, 0.25), 'k:', lw = 4)
plt.plot((588, 588), (PR_plot, 20), 'k--', lw = 2)
plt.legend(fontsize = 16)
plt.ylim(PR_plot,20)
plt.yticks(fontsize = 14)
plt.show()
fig.savefig('plots/policy_rate.pdf',dpi=300)
del fig
#
Xf = EFK_SRTSM(nparam,maturities,forwardrates,2)
rhoP = parameters[0:9];
rhoP = np.reshape(rhoP, [3,3], order='F')
muP = parameters[9:12]
rhoQ1 = np.abs(parameters[12])+ parameters[13]
rhoQ2 = parameters[13]
rhoQ = np.array([[rhoQ1, 0, 0], [0, rhoQ2, 1], [0, 0, rhoQ2]])
sigma = [[abs(parameters[14]), 0, 0],
        [parameters[15], abs(parameters[17]), 0],
        [parameters[16], parameters[18], abs(parameters[19])]]
delta0 = parameters[20]
T = len(forwardrates_time)
tau_median = np.zeros((T,1)); tau = np.zeros((10000,T))

for t in range(id_2009Jan[0], T+1):
    X0 = Xf[:,t-1]
    if delta0 + X0[0] + X0[1] < rlb :
        tau_temp = TAU_P(X0,muP,rhoP,sigma,delta0,rlb,10000)/12;
        tau_median[t-1] = np.median(tau_temp)
        
time_fwdguidance = np.array([201108, 201201, 201209, 201306])
id_fwdguidance = np.empty((len(time_fwdguidance),1))
id_fwdguidance[:] = np.NAN
for i in range(0, len(time_fwdguidance)):
    id_fwdguidance[i] = np.where(forwardrates_time == time_fwdguidance[i])[0] - id_2009Jan[0] +1

temp = np.arange(1, 61)/12.0
temp1 = tau_median[id_2009Jan[0]:T]
fig = plt.figure(5)
for i in range(0, len(time_fwdguidance)):
   plt.plot((id_fwdguidance[i], id_fwdguidance[i]), (0, 8),  'g', lw = 2)
plt.plot(temp[:, None] + temp1,'.',lw = 4,ms = 20)
plt.plot((1,61), (1/12, 61/12), 'k--')
plt.legend(['market anticipation'])
plt.ylim(0,8)
plt.xlim(1, 61)
plt.grid()
plt.show()
fig.savefig('plots/expected_lift_off.pdf',dpi=300)
del fig
# compare the extended shadow rate with event dates (Figure 13)
time = []
temp = pd.read_csv('results/SR_extend.csv', header = -1)
time = np.array(temp[0]); SR = np.array(temp[1]); 
del temp

id_2008Jan = np.where(time ==200801); time = time[id_2008Jan[0]:]; SR = SR[id_2008Jan[0]:];
time_event = np.array([200811, 201003, 201011, 201106, 201109, 201212, 201209, 201410])
id_event = np.empty((len(time_event), 1))
id_event[:] = np.NAN

for i in range(0, len(time_event)):
    id_event[i] = np.where(time == time_event[i])
    
fig = plt.figure(6)
plt.xticks(range(1, 86, 12), ['2008','2009','2010','2011','2012','2013','2014','2015'], fontsize = 14)
for i in range(0, 8):
    plt.plot((id_event[i], id_event[i]),(-3, 2.5),'g',lw=2)

p2 = plt.plot(SR,'b',lw = 4,ms=20)
plt.xlim(1,90)
plt.ylim(-3, 2.5)
plt.yticks(fontsize = 14)
plt.grid()
plt.show()
fig.savefig('plots/extended_shadowrate.pdf',dpi=300)

