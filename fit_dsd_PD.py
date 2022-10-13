#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.colors import LogNorm
import matplotlib as mpl
import matplotlib.cm as cm
from scipy import optimize
import tqdm
#%%
### read dataframe
mode = '0plot'
station = 'Anbu'
data_qc = 'yes'
stations = {'Anbu': '466910', 'Taipei': '466920', 'Keelung': '466940'}

### infile should be CWB dataframe in .csv created with 'make_dataframe.py'
infile = stations[station]+'_20210912_dataframe.csv'

name = ['Date','Time','RR(mm/h)', 'AccumR(mm)', 'WC_SYNOP_WaWa',	'WC_METAR/SPECI',	'WC_NWS',	'Reflectivity (dBz)',	'VIS(m)',	'Signal_Amp',	'Particle_Number',	'T_sensor(°C)',	'Heating current (A)',	'Sensor voltage (V)',	'KE',	'Snow(mm/h)', 'Spectrum']
class_no = np.arange(1024)
class_no.astype(str)
name = np.append(name,class_no)
# print(name.size)

### Parsivel settings
D_bin = [0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.75,3.25,3.75,4.25,4.75,5.5,6.5,7.5,8.5,9.5,11.,13.,15.,17.,19.,21.5,24.5]
V_bin = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12,13.6,15.2,17.6,20.8]
### interval of class i
dD = [.125]*10+[.25]*5+[.5]*5+[1]*5+[2]*5+[3, 3]
dV = [.1]*10+[.2]*5+[.4]*5+[.8]*5+[1.6]*5+[3.2, 3.2]
### observing time (s)
dt = 60
### sampling area of Parsivel
Fs=180*30*10**(-6)

### extract data of time window for calculation        
def set_time_window(st,et):
    data = pd.DataFrame(df[(df.index>=st) & (df.index<et)])
    return data

### get density at the observation altitude
### This is to calc Vt for data QC
def get_es(T):
    # T in C
    es = 6.112*np.exp(17.67*T/(T+243.5))
    return es
    
def get_rho(p,T):
    # p in hPa
    # T in C
    p = p*100
    T = T +273.15
    R = 287    
    rho = p/(R*T) 
    return rho

### get surface data for density retrieval
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

### calc terminal velocity
def get_Vt(x):
    # get empirical terminal velocity based on Atlas et al. (1973)
    # but with considering air density
    V_a73 =9.65-10.3*np.exp(-6.*np.array(x)/10.)*(rho0/rho)**(0.4) 
    return V_a73

V_a73 = get_Vt(D_bin)
V_a73_low = V_a73*0.5
V_a73_high = V_a73*1.5

### create mask for the first & second QC
mask = np.zeros([32, 32],dtype='bool')
for i in range(32):     # for V
    for j in range(32): # for D
        ### remove outside a73 +_50 % threshold
        if (V_bin[i] < V_a73_low[j] or V_bin[i] > V_a73_high[j]):
            mask[i][j]=True
        ### remove D < 0.2 and D > 10 
        if(D_bin[j] < 0.2 or D_bin[j] > 10.):
            mask[i][j]=True 
                        
# mask = pd.DataFrame(mask,index=D_bin,columns=V_bin) # to check mask
# mask_r = np.logical_not(mask)

### data QC
def qc_1(raw):
    raw[mask] = 0
    return raw

### read csv file
df = pd.read_csv(infile, header=None, names=name, sep=',', index_col=0,parse_dates=[[0,1]],low_memory=False)
### set time zone (LST-> UTC)
df.index = df.index.tz_localize('Asia/Taipei')
df.index = df.index.tz_convert('UTC')

#%%
def get_dsd(data):
    # get DSD spectrum --------------------------
    # output:
    #   spectrum: N(Dp) with QC done
    #   rawspec : N(Dp) raw spectrum
    #   NDi     : N(Di) converted
    #--------------------------------------------
    
    ### split out spectral data and fill with zeros as needed
    raw = data.loc[:, '0':'1023']
    raw = raw.replace('<SPECTRUM>', 0)
    raw = raw.replace('</SPECTRUM>', 0)
    raw = raw.fillna(0)

    ### initialize spectral data array
    spectrum = np.zeros([len(raw), 32, 32])
    rawspec = np.zeros([len(raw), 32, 32])
    NDi = np.zeros([len(raw), 32])
    NDi_raw = np.zeros([len(raw), 32])

    ### reshape raw spectrum data
    for i in range(0, len(raw)):
        spectrum[i] = raw.iloc[i, :].values.reshape(32, 32)
        ### keep raw data
        rawspec[i] = raw.iloc[i, :].values.reshape(32, 32)
        ### qc every 1 min data
        spectrum[i] = qc_1(spectrum[i])
        ### remove if drop number < 10/min
        num_per_diam = np.sum(spectrum[i], axis=1)
        if np.sum(num_per_diam) < 10:
            spectrum[i] = 0
        ### remove if rain rate < 0.1        
        if data.iloc[i,0] < 0.1:
            spectrum[i] = 0 
        
        ### calc QCed num_per_diam
        spec_QC = spectrum[i]
        spec_raw = rawspec[i]
        numQC = np.sum(spec_QC, axis=0)
        numraw = np.sum(spec_raw, axis=0)
        
        # spectrum = np.where(spectrum == 0.0000000e+00, np.nan, spectrum)
        
        
        ### calc NDi
        NDi[i] = numQC/(Fs*dt*V_a73*dD)
        
        ### raw NDi
        NDi_raw[i] = numraw/(Fs*dt*V_a73*dD)
                    
    return spectrum, rawspec, NDi, NDi_raw    

def get_sum_data(spectrum):
    # output:
    #   NDp          : NDp sum profile (for DSD plot) [D_bin]
    #   NV           : same as NDp but for V
    #   D_V          : sum profile for D_V plot [D_bin,V_bin]
    
    ### sum number of droplets per size class for each measurement
    num_per_diam = np.sum(spectrum, axis=1)
    num_per_v = np.sum(spectrum, axis=2)
    
    ### sum over time
    NDp = np.sum(num_per_diam, axis=0)
    NV = np.sum(num_per_v, axis=0)
    D_V = np.sum(spectrum, axis=0)
        
    ### mask 0 for plot
    NDp = np.where(NDp == 0.0000000e+00, np.nan, NDp)
    NV = np.where(NV == 0.0000000e+00, np.nan, NV)
    
    return num_per_diam, NDp, NV, D_V


def get_mean_dsd(NDi,dnum):
    NDi_sum = np.sum(NDi, axis=0)    
    ### take average
    NDi_mean = NDi_sum/dnum
    ### mask 0 for plot
    NDi_mean = np.where(NDi_mean == 0.0000000e+00, np.nan, NDi_mean)    
    return NDi_mean
    

def get_rain(data):
    rr = data[['RR(mm/h)','AccumR(mm)']]
    return rr
    

### set start_time(st) & end_time:(et) for accumulation
### analysis time
st = pd.Timestamp("2021-09-12T01:30:00.00+00:00")
et = pd.Timestamp("2021-09-12T12:00:00.00+00:00")
### 24-h (2021/09/12)
# st = pd.Timestamp("2021-09-12T00:00:00.00+00:00")
# et = pd.Timestamp("2021-09-12T23:59:00.00+00:00")
data = set_time_window(st,et)


### get spectrum & NDi [spectrum,raw,NDi]
whole = get_dsd(data)

spec_QC = whole[0]
spec_raw = whole[1]

if data_qc =='yes':
    dsd_data = whole[2]
    mode = 'QC'
elif data_qc == 'no':
    dsd_data = whole[3]
    mode = 'raw'
    
### set value 0 to Nan
dsd_data = np.where(dsd_data == 0.0000000e+00, np.nan, dsd_data)

### create dataframe of DSD for all analysis period [time,Di]
df_dsd = pd.DataFrame(dsd_data,index=data.index,columns=D_bin) 
df_dsd.to_csv('Parsivel_derived_DSD-'+station+'_'+mode+'.csv')

### set bar color for plot
if station == 'Anbu':
    color="lightpink"
    edgecolor='hotpink'
elif station == 'Keelung':
    color="lightgreen"
    edgecolor='green'
elif station == 'Taipei':
    color="skyblue"
    edgecolor='steelblue'

### set array to save derived parameters
data_param = np.zeros((len(df_dsd.index), 14))  
    
#%%
# for i in tqdm.tqdm(range(len(df_dsd.index))):
for i in range(1):
    i = 250    
    
    ### fitting data prep
    dsd = df_dsd.iloc[i]
    timelabel = df_dsd.index[i]
    ydata_r = dsd[dsd.notna()]
    ydata = np.log10(ydata_r)   # take log10
    xdata = ydata.index
    
    ### skip loop if data are empty
    if ydata.empty:
        continue
    
    ### linear fit
    def func_linear(x, a, b):
        f = a * x + b
        return f
    
    # def func_linear(x, a, b):
    #     f = a * np.exp(-b * x)
    #     return f
    
    lpopt, lpcov = optimize.curve_fit(func_linear, xdata, ydata)
    # lpopt, lpcov = optimize.curve_fit(func_linear, xdata, ydata_r)
       
    ### gamma fit
    def func_gamma(x, a, b, c):
        f = a * x**c * np.exp(-b * x)
        return  f
    
    gpopt, gpcov = optimize.curve_fit(func_gamma, xdata, ydata)
    # print(gpopt)   
         
    ### calc moment from Parsivel data
    data = dsd.rename_axis('Di').reset_index()
    data_dsd = pd.DataFrame(data,columns=['Di',dsd.name])
    data_dsd = data_dsd.rename(columns={dsd.name: 'ND'})  
    data_dsd['dD'] = dD
    data_dsd['Vt'] = V_a73
    D = data_dsd['Di'].values
    Nd = data_dsd['ND'].values
    
    def moment(n):
        data_dsd['M']=data_dsd['Di']**n*data_dsd['ND']*data_dsd['dD']
        return data_dsd['M'].sum()
    
    M3 = moment(3)
    M4 = moment(4)
    M6 = moment(6)

    ### get parameters  
    rho_w = 1e-3   
    W = np.pi*rho_w*M3/6 
    D_min = data_dsd['Di'][data_dsd['ND'].notna()].min()
    D_max = data_dsd['Di'][data_dsd['ND'].notna()].max()
    Z = 10*np.log10(M6)
    Nt = moment(0)
    
    ### The mass-weighted mean diameter (Dm)
    Dm = M4/M3
    
    ### find the median volume diameter (D0) from original data
    df_fillna = data_dsd.fillna(0)
    accm = 0
    for d in range(32):
        if accm < 0.5*M3:
            accm = accm + df_fillna['Di'][d]**3*df_fillna['ND'][d]*df_fillna['dD'][d]
            foundD0 = df_fillna['Di'][d]
        else:
            # foundD1 = df_fillna['Di'][d]
            # D0 = (foundD0+foundD1)*0.5
            D0 = foundD0
            break
 
    ### calc D0 from fit (used fit parameters)
    # linear fit
    D0_l = 3.67/lpopt[1]
    # gamma fit
    lamda = gpopt[1]
    mu = gpopt[2]
    D0_m = (mu+3.67)*Dm/(mu+4)      # eq.(6.28) Ulbrich (1983)
 
    # generalized intercept parameter Nw by Bringi et al. (2003)
    # using Dm and D0, respectively
    # also mentioned by Mismi et al. (2021)
    Nwm = 4**4*W/(np.pi*rho_w*Dm**4) 
    Nw0 = 3.67**4*W/(np.pi*rho_w*D0**4)   
    
    ### calc rain rate (mm/h) based on Brandes et al (2002)
    data_dsd['RR1']=6.*10**(-4)*np.pi*data_dsd['Di']**3*data_dsd['Vt']*data_dsd['ND']*data_dsd['dD']
    rr1 = data_dsd['RR1'].sum()
    ### calc rain rate (mm/h) based on Fukao and Hamazu (2014)
    data_dsd['RR2']=np.pi/6*data_dsd['Di']**3*data_dsd['Vt']*data_dsd['ND']*data_dsd['dD']*10**3*3600*10**(-9)
    rr2 = data_dsd['RR2'].sum()
    
    ### get Z & R from Parsivel data
    Z_pd = df.iloc[i]['Reflectivity (dBz)']
    if Z_pd < -9.:
        Z_pd = 'Nan'
    else:
        Z_pd = "%5.2f (dBZ)" % Z_pd
    
    R_pd = df.iloc[i]['RR(mm/h)']
    R_pd = "%5.2f (mm/h)" % R_pd
    
    ### save all derived parameters
    param = [Z, rr1, rr2, W, D_min, D_max, Dm, D0, D0_m,lamda,mu,Nwm,Nw0,Nt]
    data_param[i] = param
    
    
    ### ~~~ start plotting if mode is 'plot' ~~~
    if mode == 'plot':
        fig = plt.figure(figsize=(6, 6),facecolor="white")
        ax = fig.add_subplot(1, 1, 1)

        ymin=1e-1
        ymax=1e4
        # unit = ''
        unit = '$m^{-3}mm^{-1}$'

        plt.xlim([0, 7])
        plt.ylim([ymin, ymax])
        ax.set_yscale("log")
        plt.xlabel("D (mm)", fontsize=16)
        plt.ylabel("N(D) "+unit, fontsize=16)
        
        plt.bar(D_bin,dsd,width=dD,color=color,edgecolor=edgecolor,alpha=0.5)
        plt.scatter(D_bin,dsd,color="k",marker='o')
        plt.plot(D_bin,dsd,color="gray",marker='o')
        plt.title(timelabel.strftime('%Y/%m/%d %H:%M')+" UTC   DSD at "+station, fontsize=15)
        
        Di = np.array(D_bin)
        lfit = 10**(func_linear(Di, *lpopt))
        gfit = 10**(func_gamma(Di, *gpopt))
            
        plt.plot(Di, lfit, color = 'mistyrose',linestyle='--',linewidth=1.0)
        plt.plot(xdata, 10**(func_linear(xdata, *lpopt)), color = 'r',linestyle='-',linewidth=2.,
                label='Linear fit: \na=%5.3f, b=%5.3f' % tuple(lpopt))
        # plt.plot(xdata, func_linear(xdata, *lpopt), color = 'r',linestyle='-',linewidth=2.,
        #         label='Linear fit: \na=%5.3f, b=%5.3f' % tuple(lpopt))    
        
        plt.plot(Di, gfit, color = 'powderblue',linestyle='--',linewidth=1.0)
        plt.plot(xdata, 10**(func_gamma(xdata, *gpopt)), 'b',linestyle='-',linewidth=2.,
            label='Gamma fit: \n$N_0$=%5.3f, \u039b=%5.3f, \u03bc=%5.3f' % tuple(gpopt))

        plt.axvline(Dm, color=edgecolor, linestyle='--',linewidth=2.5,label='$D_m$ : %5.2f mm'% Dm)
        
        plt.text(4, 13, "Moment method", weight='bold',fontsize=13)
        plt.text(4, 10, "-----------------------", fontsize=15)
        plt.text(4, 5.5, "Z = %5.2f (dBZ)" % Z, fontsize=15)
        plt.text(4, 3, "R1 = %5.2f (mm/h)" % rr1, fontsize=15)
        plt.text(4, 1.5, "R2 = %5.2f (mm/h)" % rr2, fontsize=15)
        
        plt.text(4, 100, "from Parsivel", weight='bold',fontsize=13,color="gray")
        plt.text(4, 80, "-----------------------", fontsize=15,color="gray")    
        plt.text(4, 50, "$Z_{obs}$ = "+Z_pd, fontsize=14,color="gray")
        plt.text(4, 30, "$R_{obs}$ = "+R_pd, fontsize=14,color="gray")
        
        plt.legend(loc='upper right')   

#%%
        ### save plot
        figname = "img/DSD/1_min/fit/"+station+"_"+timelabel.strftime('%Y%m%d_%H%M')+"-ND_fitting.png"
        plt.savefig(figname, bbox_inches='tight',dpi=500)
        print('Output:'+figname)
        plt.clf()
        plt.close
#%%
exit()
### create dataframe of derived parameters
cols = ['Z','RR1','RR2','W','Dmin','Dmax','Dm','D0','D0_m','\u039b','\u03bc','Nwm','Nw0','Nt']
### set 0 => Nan for avg calc
data_param = np.where(data_param == 0.0000000e+00, np.nan, data_param)
df_param = pd.DataFrame(data_param,index=df_dsd.index, columns=cols)
### calc avg for station
mean = df_param.mean()
mean.name = station
mean['Dmin'] = df_param['Dmin'].min()
mean['Dmax'] = df_param['Dmax'].max()
print(mean.to_markdown())   # show table
Dm_avg = mean['Dm']
D0_avg = mean['D0']
D0_m_avg = mean['D0_m']
lamda_avg = mean['\u039b']
mu_avg = mean['\u03bc']

#%%
### write derived parameters     
# df_param.to_csv('Parsivel_derived_param-'+station+'_'+mode+'.csv')

# %%
# check fit
fig = plt.figure(figsize=(6, 6),facecolor="white")
ax = fig.add_subplot(1, 1, 1)
D = np.linspace(0,7,100)

a,b = lpopt

y1 = a*np.exp(-b*D)
plt.plot(D,np.log10(y1))


a,b,c = gpopt
y2 = a*D**b*np.exp(-c*D)
# plt.plot(D,np.log10(y2))

c1= 0
y3 = a*D**c1*np.exp(-b*D)

# plt.plot(D,np.log10(y3))

# ax.set_yscale("log")
# plt.plot(Di,dsd)
plt.show()


# %%
