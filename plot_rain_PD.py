#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.colors import LogNorm
import pandas.tseries.offsets as offsets
#%%
station = 'Anbu'
stations = {'Anbu': '466910', 'Taipei': '466920', 'Keelung': '466940'}
mode = 'QC'

# infile should be CWB dataframe in .csv created with 'make_dataframe.py'
infile1 = stations[station]+'_20210912_dataframe.csv'
name = ['Date','Time','RR(mm/h)', 'AccumR(mm)', 'WC_SYNOP_WaWa',	'WC_METAR/SPECI',	'WC_NWS',	'Reflectivity (dBz)',	'VIS(m)',	'Signal_Amp',	'Particle_Number',	'T_sensor(°C)',	'Heating current (A)',	'Sensor voltage (V)',	'KE',	'Snow(mm/h)', 'Spectrum']
class_no = np.arange(1024)
class_no.astype(str)
name = np.append(name,class_no)
df1 = pd.read_csv(infile1, header=None, names=name, sep=',', index_col=0,parse_dates=[[0,1]],low_memory=False)

### get surface data for hourly rain
df_sfc = pd.read_csv('../Surface/210912_'+station+'.dat',delim_whitespace=True,header=0)
df_sfc.index = pd.to_datetime(df_sfc['yyyymmddhh'],format="%Y%m%d%H")


### set time zone (LST-> UTC)
df1.index = df1.index.tz_localize('Asia/Taipei')
df1.index = df1.index.tz_convert('UTC')
df_sfc.index = df_sfc.index.tz_localize('Asia/Taipei')
df_sfc.index = df_sfc.index.tz_convert('UTC')
rr_sfc = df_sfc['PP01']


# read parsivel DSD derived rain
# infile2 = 'Parsivel_derived_param-'+station+'_'+mode+'.csv'
infile2 = 'Parsivel_derived_param-'+station+'_'+mode+'_interpDSD.csv'
df = pd.read_csv(infile2, header=0, sep=',',low_memory=False)
df['Date_Time'] = pd.to_datetime(df['Date_Time'])
df2 = df.set_index("Date_Time")

def set_time_window(df,st,et):
    data = pd.DataFrame(df[(df.index>=st) & (df.index<et)])
    return data

# set start_time(st) & end_time:(et)
# analysis time
st = pd.Timestamp("2021-09-12T01:30:00.00+00:00")
et = pd.Timestamp("2021-09-12T11:59:00.00+00:00")

rain_PD = set_time_window(df1,st,et)['RR(mm/h)']
rain_DSD = set_time_window(df2,st,et)['RR1']
#%%
# plot rain rate time seriese
fig= plt.figure(figsize=(15, 3),facecolor="white")
ax1 = fig.add_subplot(1, 1, 1)
ax1.bar(rain_PD.index, rain_PD,width=0.0005,color='deepskyblue',edgecolor='deepskyblue',label='(1) PD observed',alpha=0.5)
ax1.bar(rain_DSD.index, rain_DSD,width=0.0005,color='lightgreen',edgecolor='lightgreen',label='(2) DSD retrieved',alpha=0.5)
ax1.plot(rain_DSD.index, rain_DSD/rain_PD*100,color='orange',label='ratio (2)/(1)[%]')

### hourly data plot
ax1.plot(rr_sfc.index, rr_sfc,color='red',label='Surface hourly accum',linewidth=2,marker='o')

### calc hourly accum from Parsivel
rain_PD_H = rain_PD.resample('H').sum()/60
rain_DSD_H = rain_DSD.resample('H').sum()/60
### ... and 1h offset (adjustment for surface data)
rain_PD_H.index = rain_PD_H.index + offsets.Hour(1)
rain_DSD_H.index = rain_DSD_H.index + offsets.Hour(1)
ax1.plot(rain_PD_H,color='dodgerblue',label='PD retrieved hourly accum',linewidth=2,marker='o')
ax1.plot(rain_DSD_H,color='forestgreen',label='DSD retrieved hourly accum',linewidth=2,marker='o')

ax1.set_xlabel("Time (UTC)", fontsize=16)
ax1.set_ylabel("Rain rate (mm/h)", fontsize=16)
ax1.set_xlim([st, et])
ax1.set_ylim([0, 100])
ax1.grid()
plt.title("Rain rate (mm/h) from Parsivel at "+station+" :"+mode,fontsize=18)
fig.legend()
plt.show()
#%%
figname = "img/TimeSeriese/RR-"+station+"_"+st.strftime('%Y%m%d_%H%M')+"-"+et.strftime('%Y%m%d_%H%M')+"-"+mode+".png"
plt.savefig(figname, bbox_inches='tight',dpi=500)
print('Output:'+figname)
plt.close()
# %%
