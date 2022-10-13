#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.colors import LogNorm
import matplotlib as mpl
import matplotlib.cm as cm
#%%
# read dataframe
station = 'Anbu'
data_qc = 'yes'
stations = {'Anbu': '466910', 'Taipei': '466920', 'Keelung': '466940'}

# infile should be CWB dataframe in .csv created with 'make_dataframe.py'
infile = stations[station]+'_20210912_dataframe.csv'

name = ['Date','Time','RR(mm/h)', 'AccumR(mm)', 'WC_SYNOP_WaWa',	'WC_METAR/SPECI',	'WC_NWS',	'Reflectivity (dBz)',	'VIS(m)',	'Signal_Amp',	'Particle_Number',	'T_sensor(°C)',	'Heating current (A)',	'Sensor voltage (V)',	'KE',	'Snow(mm/h)', 'Spectrum']
class_no = np.arange(1024)
class_no.astype(str)
name = np.append(name,class_no)
# print(name.size)

# read parsivel DSD derived parameters
infile2 = 'Parsivel_derived_param-'+station+'.csv'
df_param = pd.read_csv(infile2, header=0, sep=',',low_memory=False)
df_param['Date_Time'] = pd.to_datetime(df_param['Date_Time'])
df2 = df_param.set_index("Date_Time")

# Parsivel settings
D_bin = [0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.75,3.25,3.75,4.25,4.75,5.5,6.5,7.5,8.5,9.5,11.,13.,15.,17.,19.,21.5,24.5]
V_bin = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12,13.6,15.2,17.6,20.8]
# interval of class i
dD = [.125]*10+[.25]*5+[.5]*5+[1]*5+[2]*5+[3, 3]
dV = [.1]*10+[.2]*5+[.4]*5+[.8]*5+[1.6]*5+[3.2, 3.2]
# observing time (s)
dt = 60
# sampling area of Parsivel
Fs=180*30*10**(-6)

# extract data of time window for calculation        
def set_time_window(st,et):
    data = pd.DataFrame(df[(df.index>=st) & (df.index<et)])
    return data

def get_es(T):
    # T in C
    es = 6.112*np.exp(17.67*T/(T+243.5))
    return es
    
# get density at the observation altitude
def get_rho(p,T):
    # p in hPa
    # T in C
    p = p*100
    T = T +273.15
    R = 287    
    rho = p/(R*T) 
    return rho

# get surface data for density retrieval
df_sfc = pd.read_csv('../Surface/210912_'+station+'.dat',delim_whitespace=True,header=0)
mean = df_sfc.mean(axis=0)
p = mean['PS01']
T = mean['TX01']
Td = mean['TD01']
e = get_es(T)    
es = get_es(Td)
q = 0.622*e/(p-e)
Tv = (T+273.15)*(1+0.61*q) - 273.15     # in C

rho = get_rho(p,Tv)
rho0 = 1.23     # air density of standard atmosphere

def get_Vt(x):
    # get empirical terminal velocity based on Atlas et al. (1973)
    # but with considering air density
    V_a73 =9.65-10.3*np.exp(-6.*np.array(x)/10.)*(rho0/rho)**(0.4) 
    return V_a73

V_a73 = get_Vt(D_bin)
V_a73_low = V_a73*0.5
V_a73_high = V_a73*1.5

# create mask for the first & second QC
mask = np.zeros([32, 32],dtype='bool')
for i in range(32):     # for V
    for j in range(32): # for D
        # remove outside a73 +_50 % threshold
        if (V_bin[i] < V_a73_low[j] or V_bin[i] > V_a73_high[j]):
            mask[i][j]=True
        # remove D < 0.2 and D > 10 
        if(D_bin[j] < 0.2 or D_bin[j] > 10.):
            mask[i][j]=True 
                        
mask = pd.DataFrame(mask,index=D_bin,columns=V_bin) # to check mask
# mask_r = np.logical_not(mask)

# data QC
def qc_1(raw):
    raw[mask] = 0
    return raw

# read csv file
df = pd.read_csv(infile, header=None, names=name, sep=',', index_col=0,parse_dates=[[0,1]],low_memory=False)
#%%
# set time zone (LST-> UTC)
df.index = df.index.tz_localize('Asia/Taipei')
df.index = df.index.tz_convert('UTC')

#%%
# testing cell
# st = pd.Timestamp("2021-09-12T00:00:00.00+00:00")
# et = pd.Timestamp("2021-09-12T13:00:00.00+00:00")
# data = pd.DataFrame(df[(df.index>=st) & (df.index<et)])
# # get DSD spectrum ---------------------------------------
# # split out spectral data and fill with zeros as needed
# spec_raw = data.loc[:, '0':'1023']
# spec_raw = spec_raw.replace('<SPECTRUM>', 0)
# spec_raw = spec_raw.replace('</SPECTRUM>', 0)
# spec_raw = spec_raw.fillna(0)

# # initialize spectral data array
# spectrum = np.zeros([len(spec_raw), 32, 32])
# rawspec = np.zeros([len(spec_raw), 32, 32])
# NDi = np.zeros([len(spec_raw), 32])

# # reshape raw spectrum data
# for i in range(0, len(spec_raw)):
#     spectrum[i] = spec_raw.iloc[i, :].values.reshape(32, 32)
#     rawspec[i] = spectrum[i]
#     # qc every 1 min data
#     spectrum[i] = qc_1(spectrum[i])
#     # remove if drop number < 10/min
#     num_per_diam = np.sum(spectrum[i], axis=1)
#     if np.sum(num_per_diam) < 10:
#         spectrum[i] = 0
#     # remove if rain rate < 0.1        
#     if data.iloc[i,0] < 0.1:
#         spectrum[i] = 0

#     # calc QCed num_per_diam
#     numQC = np.sum(spectrum[i], axis=1)
#     # calc NDi
#     # for j in range(32):     # for V
#     for l in range(32): # for D
#         NDi[i,l] = numQC[l]/(Fs*dt*V_a73[l]*dD[l])
# df_NDi = pd.DataFrame(NDi)

# # DSD time seriese ---
# fig = plt.figure()
# ax2 = fig.add_subplot(1, 1, 1)

# x = data.index
# y = D_bin
# x, y = np.meshgrid(x, y)

# cmap = mpl.cm.get_cmap("Spectral_r").copy() 
# plt.pcolormesh(x, y, NDi.transpose(), cmap=cmap, shading='auto',norm=LogNorm(vmin=1e-1, vmax=1e4))
# cbar=plt.colorbar(aspect=40,pad=0.08,orientation='horizontal',shrink=0.5)


#%%
# # plot D-V plot TEST ver
# dD, dV = np.meshgrid(dD, dV)
# x, y = np.meshgrid(D_bin, V_bin)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cmap = plt.get_cmap('jet').copy()
# cmap = plt.get_cmap('Spectral_r').copy()
# cmap.set_under('silver')
# ax.set_title('Parisivel Raw Data at '+station)
# ax.set_xlabel('D (mm)')
# ax.set_ylabel('V (m/s)')
# ax.set_xlim([0, 10])
# ax.set_ylim([0, 14])
# # plt.pcolormesh(x, y, raw ,cmap=cmap,norm=LogNorm(),shading='auto')
# plt.pcolormesh(x, y, S ,cmap=cmap,norm=LogNorm(),shading='auto')
# x = np.linspace(0,10,100)
# a73 =9.65-10.3*np.exp(-6.*x/10.) 
# plt.plot(x,a73,color='k')
# plt.plot(x,a73*0.5,color='k',linestyle='--')
# plt.plot(x,a73*1.5,color='k',linestyle='--')
# ax.fill_between(x, a73*0.5, a73*0, facecolor='gray', alpha=0.3)
# ax.fill_between(x, a73*1.5, a73*100, facecolor='gray', alpha=0.3)
# plt.colorbar()
# # plt.grid()
# plt.show()

#%%
def get_dsd(data):
    # get DSD spectrum --------------------------
    # output:
    #   spectrum: N(Dp) with QC done
    #   rawspec : N(Dp) raw spectrum
    #   NDi     : N(Di) converted
    #--------------------------------------------
    
    # split out spectral data and fill with zeros as needed
    spec_raw = data.loc[:, '0':'1023']
    spec_raw = spec_raw.replace('<SPECTRUM>', 0)
    spec_raw = spec_raw.replace('</SPECTRUM>', 0)
    spec_raw = spec_raw.fillna(0)

    # initialize spectral data array
    spectrum = np.zeros([len(spec_raw), 32, 32])
    rawspec = np.zeros([len(spec_raw), 32, 32])
    NDi = np.zeros([len(spec_raw), 32])

    # reshape raw spectrum data
    for i in range(0, len(spec_raw)):
        spectrum[i] = spec_raw.iloc[i, :].values.reshape(32, 32)
        # keep raw data
        rawspec[i] = spec_raw.iloc[i, :].values.reshape(32, 32)
        # qc every 1 min data
        spectrum[i] = qc_1(spectrum[i])
        # remove if drop number < 10/min
        num_per_diam = np.sum(spectrum[i], axis=1)
        if np.sum(num_per_diam) < 10:
            spectrum[i] = 0
        # remove if rain rate < 0.1        
        if data.iloc[i,0] < 0.1:
            spectrum[i] = 0 
        
        # calc QCed num_per_diam
        spec_QC = spectrum[i]
        numQC = np.sum(spec_QC, axis=0)
        
        # spectrum = np.where(spectrum == 0.0000000e+00, np.nan, spectrum)
        
        
        # calc NDi
        NDi[i] = numQC/(Fs*dt*V_a73*dD)
                    
    return spectrum, rawspec, NDi    

def get_sum_data(spectrum):
    # output:
    #   NDp          : NDp sum profile (for DSD plot) [D_bin]
    #   NV           : same as NDp but for V
    #   D_V          : sum profile for D_V plot [D_bin,V_bin]
    
    # sum number of droplets per size class for each measurement
    num_per_diam = np.sum(spectrum, axis=1)
    num_per_v = np.sum(spectrum, axis=2)
    
    # sum over time
    NDp = np.sum(num_per_diam, axis=0)
    NV = np.sum(num_per_v, axis=0)
    D_V = np.sum(spectrum, axis=0)
        
    # mask 0 for plot
    NDp = np.where(NDp == 0.0000000e+00, np.nan, NDp)
    NV = np.where(NV == 0.0000000e+00, np.nan, NV)
    
    return num_per_diam, NDp, NV, D_V


def get_mean_dsd(NDi,dnum):
    NDi_sum = np.sum(NDi, axis=0)    
    # take average
    NDi_mean = NDi_sum/dnum
    # mask 0 for plot
    NDi_mean = np.where(NDi_mean == 0.0000000e+00, np.nan, NDi_mean)    
    return NDi_mean
    

def get_rain(data):
    rr = data[['RR(mm/h)','AccumR(mm)']]
    return rr
    

# set start_time(st) & end_time:(et) for accumulation
# analysis time
st = pd.Timestamp("2021-09-12T01:30:00.00+00:00")
et = pd.Timestamp("2021-09-12T12:00:00.00+00:00")
# 24-h (2021/09/12)
# st = pd.Timestamp("2021-09-12T00:00:00.00+00:00")
# et = pd.Timestamp("2021-09-12T23:59:00.00+00:00")
data = set_time_window(st,et)

# Outer rainband 
st1 = pd.Timestamp("2021-09-12T01:35:00.00+00:00")
et1 = pd.Timestamp("2021-09-12T06:35:00.00+00:00")
data1 = set_time_window(st1,et1)

# Weak stratiform
st2 = pd.Timestamp("2021-09-12T06:35:00.00+00:00")
et2 = pd.Timestamp("2021-09-12T09:00:00.00+00:00")
data2 = set_time_window(st2,et2)

# get spectrum & NDi [spectrum,raw,NDi]
whole = get_dsd(data)
stage1 = get_dsd(data1)
stage2 = get_dsd(data2)

spec_QC = whole[0]
spec_raw = whole[1]
numQC = whole[2]
numQC = np.where(numQC == 0.0000000e+00, np.nan, numQC)
df_dsd = pd.DataFrame(numQC,index=data.index,columns=D_bin) 

#%%

# get N(D) for plot
if data_qc == 'yes' :
    num_per_diam, NDp, NV, D_V = get_sum_data(whole[0])
    num_per_diam1, NDp1, NV1, D_V1 = get_sum_data(stage1[0])
    num_per_diam2, NDp2, NV2, D_V2 = get_sum_data(stage2[0])
    
    # get NDi & data number (mins)
    NDi = whole[2]
    NDi1 = stage1[2]
    NDi2 = stage2[2]    
    # only count valid data num
    dnum = np.count_nonzero(np.sum(NDi,axis=1))
    dnum1 = np.count_nonzero(np.sum(NDi1,axis=1))
    dnum2 = np.count_nonzero(np.sum(NDi2,axis=1))
    print('--------------------')
    print('     Mean DSD')
    print('sample number check')
    print('--------------------')
    print(NDi.shape[0],NDi1.shape[0],NDi2.shape[0])
    print(dnum,dnum1,dnum2)
    print('--------------------')

    # get mean DSD
    NDi_mean = get_mean_dsd(NDi, dnum)
    NDi_mean1 = get_mean_dsd(NDi1, dnum1)
    NDi_mean2 = get_mean_dsd(NDi2, dnum2)
    
    # setting for plot 
    mode = 'Mean'         
    unit = '$m^{-3}mm^{-1}$'
    ymin = 1e-4
    ymax = 1e4      
    loc = 'img'
elif data_qc == 'no' :
    num_per_diam, NDp, NV, D_V = get_sum_data(whole[1])
    num_per_diam1, NDp1, NV1, D_V1= get_sum_data(stage1[1])
    num_per_diam2, NDp2, NV2, D_V2,= get_sum_data(stage2[1])
    
    # get NDi & data number (mins)
    NDi = num_per_diam
    NDi1 = num_per_diam1
    NDi2 = num_per_diam2
        
    NDi_mean = NDp
    NDi_mean1 = NDp1
    NDi_mean2 = NDp2  
      
    # setting for plot
    mode = 'Accumulated'   
    unit = 'droplets'
    ymin = 1e0
    ymax = 1e5       
    loc = 'img/raw_plot'
else :
    print('dataQC mode should be either "yes" or "no"')
    exit()

 
# get rain rate
rr = get_rain(data)
rr1 = rr[['RR(mm/h)']]

# %%
# ~~~ plot D-V plot ~~~
x, y = np.meshgrid(D_bin, V_bin)

fig = plt.figure()
ax = fig.add_subplot(111)
cmap = plt.get_cmap('jet').copy()
cmap = plt.get_cmap('Spectral_r').copy()
cmap.set_under('silver')
ax.set_title('Parisivel Raw Data at '+station)
ax.set_xlabel('D (mm)')
ax.set_ylabel('V (m/s)')
ax.set_xlim([0, 10])
ax.set_ylim([0, 14])
plt.pcolormesh(x, y, D_V,cmap=cmap,norm=LogNorm(),shading='auto')
x = np.linspace(0,20,100)
a73 =9.65-10.3*np.exp(-6.*x/10.) 
a73_mod = get_Vt(x)
plt.plot(x,a73,color='gray')
plt.plot(x,a73*0.5,color='gray',linestyle='--')
plt.plot(x,a73*1.5,color='gray',linestyle='--')

plt.plot(x,a73_mod,color='k')
plt.plot(x,a73_mod*0.5,color='k',linestyle='--')
plt.plot(x,a73_mod*1.5,color='k',linestyle='--')
ax.fill_between(x, a73_mod*0.5, a73_mod*0, facecolor='gray', alpha=0.3)
ax.fill_between(x, a73_mod*1.5, a73_mod*100, facecolor='gray', alpha=0.3)
plt.colorbar()
# plt.show()

# ~~~ plot box-whisker plot ~~~
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(x,a73_mod,color='k')
# plt.plot(x,a73_mod*0.5,color='k',linestyle='--')
# plt.plot(x,a73_mod*1.5,color='k',linestyle='--')

dataD_V = pd.DataFrame(D_V,index=D_bin,columns=V_bin)
data_V = dict()
# data_V = []
for i in range(32):
    list_V = []    
    for j in range(32):
        list_V = list_V+[float(V_bin[j])]*int(dataD_V.iloc[j,i])
        # print(list_V)
    data_V[i] = list_V
    # print(len(list_V))
    position = D_bin[i]
    ax.boxplot(list_V,positions=[position],showfliers=False,#flierprops=dict(markersize=1),
               showmeans=True,medianprops=dict(color='k'),
               whiskerprops=dict(color='k'),
               meanprops=dict(color='k'))     
    # data_V = data_V+list_V
 
# ax.set_xlim([0, 10])
# ax.set_ylim([0, 14])
ax.set_xticks(np.arange(0,11))
ax.set_xticklabels(np.arange(0,11))
plt.show()

# plt.grid()
#%%
figname = loc+"/D-V_plot/"+station+"_"+st.strftime('%Y%m%d_%H%M')+"-"+et.strftime('%Y%m%d_%H%M')+"-mod.png"
plt.savefig(figname, bbox_inches='tight',dpi=500)
print('Output:'+figname)
plt.close

#%%
# ~~~ plot DSD ~~~
# plot D-ND diagram 
fig= plt.figure(figsize=(6, 6))
plt.title(mode+" Dropsize Distribution (DSD) at "+station, fontsize=14)

Di =  D_bin

plt.semilogy(D_bin, NDi_mean, color='k', lw=2,linestyle='-',label='Total')
plt.semilogy(D_bin, NDi_mean1, color='hotpink', linestyle='-',label='stage 1')
plt.semilogy(D_bin, NDi_mean2, color='mediumpurple', linestyle='-',label='stage 2')
plt.xlim([0, 7])
plt.ylim([ymin, ymax])

plt.xlabel("D (mm)", fontsize=16)
plt.ylabel("N(D) "+unit, fontsize=16)
plt.legend()
# figname = loc+"/DSD/"+station+"_"+st.strftime('%Y%m%d_%H%M')+"-"+et.strftime('%Y%m%d_%H%M')+".png"
figname = loc+"/DSD/"+station+"_"+st.strftime('%Y%m%d_%H%M')+"-"+et.strftime('%Y%m%d_%H%M')+"-NDi_test.png"
plt.savefig(figname, bbox_inches='tight')
print('Output:'+figname)
plt.close

# exit()

#%%
# DSD time seriese ---
fig= plt.figure(figsize=(15, 5))
# fig= plt.figure(figsize=(12, 6))
# ax2 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(1, 1, 1)

x = data.index
y = Di
x, y = np.meshgrid(x, y)

cmap = mpl.cm.get_cmap("Spectral_r").copy() 
# cmap = mpl.cm.get_cmap("jet").copy() 

# plt.pcolormesh(x, y, NDi.transpose(), cmap=cmap, shading='nearest',norm=LogNorm())
plt.pcolormesh(x, y, NDi.transpose(), cmap=cmap, shading='auto',norm=LogNorm(vmin=1e-0, vmax=1e4))
cbar=plt.colorbar(aspect=40,pad=0.08,orientation='horizontal',shrink=0.5)
# cbar.set_label('$log_{10}N(D)$', fontsize=14) 
cbar.set_label('N(D) '+unit, fontsize=14,labelpad=-1.) 
ax2.set_title("Time seriese of DSD at "+station, fontsize=15)
# ax2.set_xlabel("Time (UTC)", fontsize=16)
ax2.set_ylabel("D (mm)", fontsize=16)
ax2.set_xlim([st, et])
ax2.set_ylim([0, 6])

# plot total number (log10)
# get total number in log10
num_sum = np.sum(num_per_diam, axis=1)
num_sum_log10 = np.log10(num_sum)
ax2.plot(data.index, num_sum_log10, color='navy',label='$log_{10}(N_{total})$',linewidth='2')
# plot Dm (PD derived)
ax2.plot(data.index, df2['Dm'], color='k',label='$D_{m} (mm)$',linewidth='2')
ax2.legend()
plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.9)

#%%
# figname = loc+"/TimeSeriese/DSD-"+station+"_"+st.strftime('%Y%m%d_%H%M')+"-"+et.strftime('%Y%m%d_%H%M')+".png"
figname = loc+"/TimeSeriese/DSD-"+station+"_"+st.strftime('%Y%m%d_%H%M')+"-"+et.strftime('%Y%m%d_%H%M')+"-NDi_test.png"
plt.savefig(figname, bbox_inches='tight')
print('Output:'+figname)
plt.close

# plot rain rate time seriese
fig= plt.figure(figsize=(15, 3))
ax1 = fig.add_subplot(1, 1, 1)
ax1.bar(rr.index, rr['RR(mm/h)'],width=0.0005,edgecolor='skyblue')
# ax1.plot(rr.index, rr['AccumR(mm)'],color='tomato')
ax1.set_xlabel("Time (UTC)", fontsize=16)
ax1.set_ylabel("Rain rate (mm/h)", fontsize=16)
ax1.set_xlim([st, et])
ax1.set_ylim([0, 100])
ax1.grid()

# ax1.bar(rr.index, rr1['RR(mm/h)'],width=0.0005,edgecolor='hotpink')
# ax1.bar(rr.index, rr2['RR(mm/h)'],width=0.0005,edgecolor='mediumpurple')
#%%
figname = loc+"/TimeSeriese/RR-"+station+"_"+st.strftime('%Y%m%d_%H%M')+"-"+et.strftime('%Y%m%d_%H%M')+".png"
plt.savefig(figname, bbox_inches='tight')
print('Output:'+figname)
plt.close

#%%
# ~~~ plot DSD bargraph~~~
# plot D-ND bargraph 
# bar color
color="lightgreen"
edgecolor='green'

for i in range(len(df_dsd.index)):
    fig = plt.figure(figsize=(6, 6),facecolor="white")
    ax = fig.add_subplot(1, 1, 1)

    Di =  D_bin
    ymin=1e-1
    ymax=1e4
    # unit = ''
    plt.xlim([0, 7])
    plt.ylim([ymin, ymax])
    ax.set_yscale("log")
    plt.xlabel("D (mm)", fontsize=16)
    plt.ylabel("N(D) "+unit, fontsize=16)
    
    dsd = df_dsd.iloc[i]
    timelabel = df_dsd.index[i]
    plt.bar(D_bin,dsd,width=dD,color=color,edgecolor=edgecolor)
    plt.plot(D_bin,dsd,color="k",marker='o',linewidth=2.)
    plt.title(timelabel.strftime('%Y/%m/%d %H:%M')+" UTC   DSD at "+station, fontsize=15)
    figname = loc+"/DSD/1_min/"+station+"_"+timelabel.strftime('%Y%m%d_%H%M')+"-ND.png"
    plt.savefig(figname, bbox_inches='tight',dpi=500)
    print('Output:'+figname)
    plt.clf()
    plt.close

exit()
#%%
# ~~~ plot D-NV diagram ~~~
fig= plt.figure(figsize=(6, 6))
plt.semilogy(D_bin, NV, color='limegreen', linestyle=':')
plt.xlim([D_bin[0], D_bin[31]])
plt.ylim([10**(-1), 10**(5.1)])
# drop 1st & 2nd class data (low SNR)
NV_QC = NV
NV_QC[:2] = np.nan
plt.semilogy(D_bin, NV_QC, color='limegreen')
plt.xlabel("V (m/s)", fontsize=16)
plt.ylabel("N(D)", fontsize=16)

plt.close

#%%
# calc parameters