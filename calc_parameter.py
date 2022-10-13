#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.colors import LogNorm
from scipy import optimize
import tqdm
from scipy import interpolate
#%%
station = 'Keelung'
mode = 'raw'
# read parsivel DSD ".csv" is created by "fit_dsd_PD.py"
infile = 'Parsivel_derived_DSD-'+station+'_'+mode+'.csv'
df = pd.read_csv(infile, header=0, sep=',',low_memory=False)
df['Date_Time'] = pd.to_datetime(df['Date_Time'])
df_dsd = df.set_index("Date_Time")

### Parsivel setting
D_bin = [0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.75,3.25,3.75,4.25,4.75,5.5,6.5,7.5,8.5,9.5,11.,13.,15.,17.,19.,21.5,24.5]
dD = [.125]*10+[.25]*5+[.5]*5+[1]*5+[2]*5+[3, 3]

#%% interpolation of D & N(D)
def interp(D,ND):
    ### assime dd = 0.125
    # d = np.linspace(min(D_bin), max(D_bin), 100)
    d = np.arange(min(D_bin), max(D_bin), 0.125)
    
    ### fit in log scale
    Nd = np.log10(ND)  
    fitted_curve = interpolate.interp1d(D, Nd)
    
    ### plot to check interpolation result
    # plt.scatter(D, Nd, label="observed")
    # plt.plot(d, fitted_curve(d), c="red", marker='.',label="interpolated")
    # # plt.title()
    # plt.xlabel("$D_m$ (mm)", fontsize=10)
    # plt.ylabel("$log_{10}N(D)$", fontsize=10)
    # plt.xlim([0, 6])   
    # plt.grid()
    # plt.legend()
    # plt.show()
    nd = fitted_curve(d)
    nd = 10**(nd)
    return d,nd


# plt.plot(d,nd)
#%%
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


### set array to save derived parameters
data_param = np.zeros((len(df_dsd.index), 11))  

#%%
# for i in tqdm.tqdm(range(len(df_dsd.index))):
for i in range(1):
    i = 555    
    
    dsd = df_dsd.iloc[i]
    timelabel = df_dsd.index[i]
    ydata = dsd[dsd.notna()]

    ### skip loop if data are empty
    if ydata.empty:
        continue
    
    ### interpolation
    dsd = df_dsd.iloc[i]
    D = dsd.index.astype(float)
    ND = dsd.fillna(0).values
    d,nd = interp(D,ND) 
    nd = np.where(nd == 0.0000000e+00, np.nan, nd)
    ### terminal velocity    
    V_a73 = get_Vt(d)   
    
    # ### calc moment from Parsivel data
    # data = dsd.rename_axis('Di').reset_index()
    # data_dsd = pd.DataFrame(data,columns=['Di',dsd.name])
    # data_dsd = data_dsd.rename(columns={dsd.name: 'ND'})  
    # data_dsd['dD'] = dD
    # data_dsd['Vt'] = V_a73
    # D = data_dsd['Di'].values
    # Nd = data_dsd['ND'].values
    
    ### calc moment from interpolated DSD data
    data_dsd = pd.DataFrame(data={'Di': d, 
          'ND': nd,
          'dD': [0.125]*len(d),
          'Vt': V_a73}) 

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
    D0 = 0
    h_W = 0.5*M3
    # print("W/2: "+str(h_W))
    for dd in range(len(d)):
        if accm < h_W:
            accm = accm + df_fillna['Di'][dd]**3*df_fillna['ND'][dd]*df_fillna['dD'][dd]
            foundD0 = df_fillna['Di'][dd]
            D0 = foundD0
            # print("AccumM3: "+str(accm))
        else:
            # print(accm,h_W)
            foundD1 = df_fillna['Di'][dd]
            D0 = (foundD0+foundD1)*0.5
            # D0 = foundD0
            break

    ### calc D0 from fit (used fit parameters)
    # # linear fit
    # D0_l = 3.67/lpopt[1]
    # # gamma fit
    # lamda = gpopt[1]
    # mu = gpopt[2]
    # D0_m = (mu+3.67)*Dm/(mu+4)      # eq.(6.28) Ulbrich (1983)

    ### generalized intercept parameter Nw by Bringi et al. (2003)
    ### using Dm and D0, respectively
    ### also mentioned by Mismi et al. (2021)
    Nwm = 4**4*W/(np.pi*rho_w*Dm**4) 
    Nw0 = 3.67**4*W/(np.pi*rho_w*D0**4)   

    ### calc rain rate (mm/h) based on Brandes et al (2002)
    data_dsd['RR1']=6.*10**(-4)*np.pi*data_dsd['Di']**3*data_dsd['Vt']*data_dsd['ND']*data_dsd['dD']
    rr1 = data_dsd['RR1'].sum()
    ### calc rain rate (mm/h) based on Fukao and Hamazu (2014)
    data_dsd['RR2']=np.pi/6*data_dsd['Di']**3*data_dsd['Vt']*data_dsd['ND']*data_dsd['dD']*10**3*3600*10**(-9)
    rr2 = data_dsd['RR2'].sum()
    
    ### save all derived parameters
    param = [Z, rr1, rr2, W, D_min, D_max, Dm, D0,Nwm,Nw0,Nt]
    data_param[i] = param
    
    
# %%
### create dataframe of derived parameters
cols = ['Z','RR1','RR2','W','Dmin','Dmax','Dm','D0','Nwm','Nw0','Nt']
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
#%%
### write derived parameters     
df_param.to_csv('Parsivel_derived_param-'+station+'_'+mode+'_interpDSD.csv')

# %%
