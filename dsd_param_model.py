#%%
import numpy as np
import matplotlib.pyplot as plt
from math import gamma
from matplotlib import cm
import pandas as pd
import subprocess
#%%
# modified gamma distribution
N0 = 8.0*10**(6)
W = 1.0
D0_mm = 2.0
D0_m = D0_mm*0.001
D0 = D0_mm
rho_w = 1e-3    #10**6
D = np.linspace(0,5,101)
# D = D*0.001

# m = -2.

#%%
# following eq. (6.25) of Fukao & Hamatsu (2005)
def ND_gamma(m):
    ND_gamma = N0*D**m*np.exp(-(m+3.67)/D0*D)
    return ND_gamma

fig= plt.figure(figsize=(5, 5))
plt.semilogy(D, ND_gamma(-2), color='limegreen', linestyle=':',label='\u03bc = -2')
plt.semilogy(D, ND_gamma(0), color='limegreen', linestyle='--',label='\u03bc = 0')
plt.semilogy(D, ND_gamma(2), color='limegreen', linestyle='-',label='\u03bc = 2')
plt.legend()
plt.show()
plt.close()

# %%
# test mu

# m = [-2,-1,0,1,2]
# m = np.linspace(0,1,11)
# np.round(m,decimals=1)
m = np.arange(-3,5)

def calc_lamda(m):
    lamda = (m+3.67)/D0   
    return lamda
       
def calc_N0(m):
    # N0 = (6.*W*(m+3.67)**(m+4))/(rho_w*np.pi)/(gamma(m+4)*D0**(m+4))
    N0 = (6.*W*calc_lamda(m)**(m+4))/(rho_w*np.pi*gamma(m+4))   #*10**6
    return N0


def moment(m,n):
    lamda = calc_lamda(m)   
    M = calc_N0(m)*gamma(n+m+1)/lamda**(n+m+1)
    return M
    
def calc_Dm(m):
    Dm = D0*(m+4)/(m+3.67)
    return "{:.2f}".format(Dm) 

def ND_gamma0(m):
    N0 = calc_N0(m)
    ND_gamma = N0*D**m*np.exp(-(m+3.67)*D/D0)    
    return ND_gamma

def calc_Nt(m):
    # following Ulbrich (1983) eq.(15)
    Nt = gamma(1+m)*calc_N0(m)*D0**(1+m)/(3.67+m)**(1+m)
    return Nt

def calc_Z(m):
    N0 = calc_N0(m)
    print('N0 ='+str(N0))   
    # get Z
    zh = moment(m,6)
    dbz = 10*np.log10(zh)
    print('Z = '+"{:.2f}".format(dbz)+' dBZ')
    return "{:.1f}".format(dbz)
    
# def calc_R(m):
#     lamda = calc_lamda(m)   
#     R = np.pi*calc_N0(m)*(9.65*(gamma(4+m)/lamda**(4+m))-10.3*(gamma(4+m)/(lamda+600)**(4+m)))/6
#     print('R = '+"{:.2f}".format(R)+' mm/h')
#     return "{:.3f}".format(R)


#%% test rain
def calc_RR(m):
    # below are in d [m]
    d = D*10**(-3)
    rho_w = 10**6

    def calc_lamda(m):
        lamda = (m+3.67)/D0_m   
        return lamda
        
    def calc_N0(m):
        N0 = (6.*W*calc_lamda(m)**(m+4))/(rho_w*np.pi*gamma(m+4))   
        print('N0 = %5.2f'% N0)
        return N0
        
    def calc_R(m):
        lamda = calc_lamda(m)   
        RR = np.pi*calc_N0(m)*(9.65*(gamma(4+m)/lamda**(4+m))-10.3*(gamma(4+m)/(lamda+600)**(4+m)))/6
        R = RR*(3.6*10**6)
        print('R = '+"{:.2f}".format(R)+' mm/h')
        return "{:.2f}".format(R)

    # ### calc terminal velocity
    # def get_Vt(x):
    #     # get empirical terminal velocity based on Atlas et al. (1973)
    #     # but with considering air density
    #     V_a73 =9.65-10.3*np.exp(-600.*np.array(x))
    #     return V_a73
    
    return calc_R(m)

# m=2
# lamda = calc_lamda(m) 
# lamda = (m+3.67)/D0_m  
# N0_m =calc_N0(m)
# R = np.pi*N0_m*(9.65*(gamma(4+m)/lamda**(4+m))-10.3*(gamma(4+m)/(lamda+600)**(4+m)))/6
# ### conversion to (mm/h)
# R = R*(3.6*10**6)

# accumulation
# dd = d[1]-d[0]
# RR = sum(np.pi*d**3*ND_gamma0(m)*get_Vt(d)*dd)
# # RR = sum(np.pi*d**3*ND_gamma0(m)*get_Vt(d))
# R = RR/(3.6*10**6)
# R = RR*(3.6*10**6)
# print(RR)
# ZZ = calc_Z(m)
# rr = calc_RR(m)

#%%

# plot DSD

fig= plt.figure(figsize=(5, 5),facecolor="white")
colors = plt.cm.rainbow(np.linspace(0,1,len(m)))

# for table
data_zr = np.zeros(shape=[len(m),3])
df_zr = pd.DataFrame(data_zr,index=m,columns=['Z (dBZ)','R (mm/h)','$D_{m}$ (mm)'])

j = 0

for i in m:
    print('------- \u03bc = %3.1f'  % i+'------')
    # calc_Z(i)
    # calc_R(i)
    
    df_zr.loc[i]['Z (dBZ)']=calc_Z(i)
    df_zr.loc[i]['R (mm/h)']=calc_RR(i)
    df_zr.loc[i]['$D_{m}$ (mm)']=calc_Dm(i)
    lc = colors[j]
    j +=1    
    # lc = colors[i+3]
    plt.xlim([0, 5])

    plt.semilogy(D, ND_gamma0(i), color=lc, linestyle='-',label='\u03bc = %3.1f'  % i)
    plt.axvline(float(calc_Dm(i)), color=lc, linestyle=':',linewidth=1.)
    

# plt.semilogy(D, ND_gamma0(-2), color='dodgerblue', linestyle=':',label='\u03bc = -2')
# plt.semilogy(D, ND_gamma0(0), color='dodgerblue', linestyle='--',label='\u03bc = 0')
# plt.semilogy(D, ND_gamma0(2), color='dodgerblue', linestyle='-',label='\u03bc = 2')

plt.title('W = '+str(W)+' g$m^{-3}$,  $D_{0}$= '+str(D0)+' mm', fontsize=14)
ymin = 1e-1
ymin = 1e0
ymax = 1e4
unit = '$m^{-3}mm^{-1}$'
plt.xlim([0, 5])
plt.ylim([ymin, ymax])
plt.xlabel("D (mm)", fontsize=10)
plt.ylabel("N(D) "+unit, fontsize=10)
plt.legend(loc='upper right')
# plt.ylim([10**(3), 10**(7)])
plt.show()
# %%
figname1 = "img/Gamma_model_DSD_W"+str(W)+"-D0_"+str(D0)+".png"
plt.savefig(figname1, bbox_inches='tight',dpi=500)
print('Output:'+figname1)
# %%
# plot table
print(df_zr.to_markdown())

fig = plt.figure(figsize=(4, 5),facecolor="white")
plt.title('W = '+str(W)+' g$m^{-3}$,  $D_{0}$= '+str(D0)+' mm\n', fontsize=15)
plt.axis('off')
# plt.axis('tight')
# ax.table(cellText=df_zr.values,
#          colLabels=df_zr.columns,
#          loc='center',
#          bbox=[0,0,1,1])

df_zr = df_zr.rename_axis('\u03bc').reset_index()
table = plt.table(cellText=df_zr.values,
         colLabels=['\u03bc','Z (dBZ)', 'R (mm/h)','$D_{m}$ (mm)'],
         loc='center',
         cellLoc='center',
         colColours=["gainsboro"]*4,
         bbox=[0,0,1,1])
table.set_fontsize(15)

# color cell
alpha = 0.2
colors[:,-1] = alpha
for i in range(len(m)):
    table[(i+1,0)].set_facecolor(colors[i])
# table[(0, 0)].set_facecolor("lightgray")

plt.show()
figname2 = "img/Gamma_model_DSD_W"+str(W)+"-D0_"+str(D0)+"table.png"
plt.savefig(figname2,bbox_inches='tight',dpi=500)

# %%
# conbine 2 figs
fignameo = "img/Gamma_model_DSD_W"+str(W)+"-D0_"+str(D0)+"-combined-modified.png"
cmd = 'convert +append '+figname1+' '+figname2+' '+fignameo
subprocess.run(cmd, shell=True)
print('Output:'+fignameo)

#%%
fig = plt.figure(figsize=(5, 5),facecolor="white")
ratio = df_zr['$D_{m}$ (mm)']/D0
plt.plot(m,ratio)
plt.ylim([0, 2])
plt.xlabel("\u03bc", fontsize=10)
plt.ylabel("$D_{m}$/$D_{0}$ ", fontsize=10)
plt.axhline(1, color='k', linestyle=':',linewidth=1.)
plt.savefig('img/Gamma_model/Fig4_Ulbrich.png',bbox_inches='tight',dpi=500)

# %%
