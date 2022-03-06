# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 17:59:05 2020

@author: Egecan
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:52:25 2019

@author: Egecan
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import argrelextrema
from scipy.signal import lfilter
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

FONTSIZE=16
#Start=825
Start=650

def linear(x,a,b):
    return a*x+b

def quadratic(x,a,b,c):
    return a*x*x+b*x+c

def substract_line(ELY_array): #ELY_array is either EY or LY array. ELY is a string to tell the difference. 
    
    
    """Short Linear"""
    
    ELY_array=np.asfarray(ELY_array)
    start=0
    stop=300
    ELY_array=ELY_array*(1.0/ELY_array[stop])


    length=np.size(ELY_array[0])
    xline=np.linspace(810,840,length)
    
    popt0, pcov0 = curve_fit(linear, xline[start:stop],ELY_array[start:stop])

    
    ELY_array=ELY_array-linear(xline, *popt0)

    # ELY_array[1]=ELY_array[1]*(ELY_array[0][EY_peaks[0]]/ELY_array[1][EY_peaks[1]])
       
    """Long Linear"""
    
    # ELY_array=np.asfarray(ELY_array)
    # start=0
    # stop=540
    # ELY_array[0]=ELY_array[0]*(1.0/ELY_array[0][stop])
    # ELY_array[1]=ELY_array[1]*(1.0/ELY_array[1][stop])

    # length=np.size(ELY_array[0])
    # xline=np.linspace(650,750,length)
    
    # popt0, pcov0 = curve_fit(linear, xline[start:stop],ELY_array[0][start:stop])
    # popt1, pcov1 = curve_fit(linear, xline[start:stop],ELY_array[1][start:stop])
    
    # ELY_array[0]=ELY_array[0]-linear(xline, *popt0)
    # ELY_array[1]=ELY_array[1]-linear(xline, *popt1)

    # ELY_array[1]=ELY_array[1]*(ELY_array[0][EY_peaks[0]]/ELY_array[1][EY_peaks[1]]) 
    
    
    """Short Quadratic"""

    # start=360
    # stop=560
    # ELY_array[0]=ELY_array[0]*(1.0/ELY_array[0][stop])
    # ELY_array[1]=ELY_array[1]*(1.0/ELY_array[1][stop])

    # length=np.size(ELY_array[0])
    # xline=np.linspace(650,750,length)
    
    # popt0, pcov0 = curve_fit(quadratic, xline[start:stop],ELY_array[0][start:stop])
    # popt1, pcov1 = curve_fit(quadratic, xline[start:stop],ELY_array[1][start:stop])
    
    # ELY_array[0]=ELY_array[0]-quadratic(xline, *popt0)
    # ELY_array[1]=ELY_array[1]-quadratic(xline, *popt1)

    # ELY_array[1]=ELY_array[1]*(ELY_array[0][EY_peaks[0]]/ELY_array[1][EY_peaks[1]])

    
    """Long Quadratic"""

    # start=0
    # stop=560
    # ELY_array[0]=ELY_array[0]*(1.0/ELY_array[0][stop])
    # ELY_array[1]=ELY_array[1]*(1.0/ELY_array[1][stop])

    # length=np.size(ELY_array[0])
    # xline=np.linspace(650,750,length)
    
    # popt0, pcov0 = curve_fit(quadratic, xline[start:stop],ELY_array[0][start:stop])
    # popt1, pcov1 = curve_fit(quadratic, xline[start:stop],ELY_array[1][start:stop])
    
    # ELY_array[0]=ELY_array[0]-quadratic(xline, *popt0)
    # ELY_array[1]=ELY_array[1]-quadratic(xline, *popt1)

    # ELY_array[1]=ELY_array[1]*(ELY_array[0][EY_peaks[0]]/ELY_array[1][EY_peaks[1]])

    
    """Linear Edge to Edge"""
    
    # interval=10
    # length=np.size(ELY_array[0])
    # line0=np.linspace(np.average(ELY_array[0][:interval]),np.average(ELY_array[0][-1*interval:]),length)
    # line1=np.linspace(np.average(ELY_array[1][:interval]),np.average(ELY_array[1][-1*interval:]),length)

    # ELY_array[0]=ELY_array[0]-line0
    # ELY_array[1]=ELY_array[1]-line1
    
    
    # if ELY_array[0][180]<0:
    #   ELY_array[0]=ELY_array[0]*-1
    # if ELY_array[1][180]<0:
    #   ELY_array[1]=ELY_array[1]*-1
    
    return (ELY_array)

def custom_plot(energy_array,EY_array,title,save_path,plot,save,show): 
    
    peak_start=595
    peak_stop=610
    
    # EY_peak_P=peak_start+peak_locator(EY_array[0][peak_start:peak_stop])[0][0]
    # EY_peak_M=peak_start+peak_locator(EY_array[1][peak_start:peak_stop])[0][0]
    EY_peaks=[[100,100],[100,100]]
    
    """Plotter function:"""
    EY_array=substract_line(EY_array,EY_peaks,"EY")
    
    # EY_peak_P=peak_start+peak_locator(EY_array[0][peak_start:peak_stop])[0][0]
    # EY_peak_M=peak_start+peak_locator(EY_array[1][peak_start:peak_stop])[0][0]
    # EY_peaks=[EY_peak_P,EY_peak_M]

    # print(EY_peaks)
    
    
    # peak_start=718
    # peak_stop=728
    # h_a=peak_start+peak_locator(EY_array[0][peak_start:peak_stop])[0][0]
    # peak_start=730
    # peak_stop=743
    # h_b=peak_start+peak_locator(EY_array[0][peak_start:peak_stop])[0][0]
    # print(h_a,h_b)
    # EY_peaks=[h_a,h_b]
    # v_a=peak_start+peak_locator(EY_array[1][peak_start:peak_stop])[0][0]
    # v_b=peak_start+peak_locator(EY_array[1][peak_start:peak_stop])[0][0]
    

    
 
    if plot==True: #BOTH SPECTRA M+ & M-
        f=plt.figure()
        ax = f.add_subplot(111)
        plt.plot(energy_array[1],EY_array[1],"-",linewidth=1,label="M-",color='orangered')
        plt.plot(energy_array[0],EY_array[0],"-",linewidth=1,label="M+",color='navy')
        plt.plot(energy_array[1][EY_peaks[1]],EY_array[1][EY_peaks[1]],"1",markersize=5,color='orangered')
        plt.plot(energy_array[1][EY_peaks[0]],EY_array[1][EY_peaks[0]],"1",markersize=5,color='orangered')
        
        plt.plot(energy_array[0][EY_peaks[0]],EY_array[0][EY_peaks[0]],"2",markersize=5,color='navy')
        plt.plot(energy_array[0][EY_peaks[1]],EY_array[0][EY_peaks[1]],"2",markersize=5,color='navy')


        plt.xlabel('Energy (eV)',fontsize=FONTSIZE-1)
        plt.ylabel('Electron Yield (Counts)' ,fontsize=FONTSIZE-1)
        plt.title("XMCD - EY vs. Energy"+" - "+title,fontsize=FONTSIZE)
        plt.text(0.1, 0.9,"M+: "+str(np.round(EY_array[0][EY_peaks[0]],6))+"\nM-: "+str(np.round(EY_array[1][EY_peaks[1]],6)) ,transform=ax.transAxes)
        plt.legend(fontsize="large")
        plt.tight_layout()
    if save==True:    
        plt.savefig(save_path+"EY - "+title+".png",dpi=600)
    if show==False:
        plt.close()
    
    peak_start=550
    peak_stop=700
    if False: #DIFFERENCE OF SPECTRA M+ & M-
        plt.figure()
        plt.plot(energy_array[0],EY_array[0]-EY_array[1],"-",markersize=1,label="M+")
        plt.plot(Start+peak_start*0.1+0.1*np.argmin((EY_array[0]-EY_array[1])[peak_start:peak_stop]),np.min((EY_array[0]-EY_array[1])[peak_start:peak_stop]),"1",markersize=5)
        plt.plot(Start+peak_start*0.1+0.1*np.argmax((EY_array[0]-EY_array[1])[peak_start:peak_stop]),np.max((EY_array[0]-EY_array[1])[peak_start:peak_stop]),"2",markersize=5)
#        plt.plot(energy_array[0][EY_peaks[0]],EY_array[0][EY_peaks[0]],"2",markersize=5,color='navy')
#        plt.plot(energy_array[1][EY_peaks[1]],EY_array[1][EY_peaks[1]],"1",markersize=5,color='orangered')
        plt.xlabel('Energy (eV)',fontsize=FONTSIZE-1)
        plt.ylabel('Electron Yield (Counts)' ,fontsize=FONTSIZE-1)
        plt.title("XMCD - EY vs. Energy"+" - Difference - "+title,fontsize=FONTSIZE)
        plt.legend(fontsize="large")
#        plt.ylim((-0.5,0.5))
        plt.tight_layout()
    if save==True:    
        plt.savefig(save_path+"EY - Difference2 "+title+".png",dpi=600)
    if show==False:
        plt.close()
   
    if False: #SUM OF SPECTRA M+ & M-
        plt.figure()
        plt.plot(energy_array[0],EY_array[0]+EY_array[1],"-",markersize=1,label="M+")
        
        plt.plot(Start+peak_start*0.1+0.1*np.argmax((EY_array[0]+EY_array[1])[peak_start:peak_stop]),np.max((EY_array[0]+EY_array[1])[peak_start:peak_stop]),"2",markersize=5)
#        plt.plot(energy_array[0][EY_peaks[0]],EY_array[0][EY_peaks[0]],"2",markersize=5,color='navy')
#        plt.plot(energy_array[1][EY_peaks[1]],EY_array[1][EY_peaks[1]],"1",markersize=5,color='orangered')
        plt.xlabel('Energy (eV)',fontsize=FONTSIZE-1)
        plt.ylabel('Electron Yield (Counts)' ,fontsize=FONTSIZE-1)
        plt.title("XMCD - EY vs. Energy"+" - Sum - "+title,fontsize=FONTSIZE)
        plt.legend(fontsize="large")
#        plt.ylim((-0.5,0.5))
        plt.tight_layout()
    if save==True:    
        plt.savefig(save_path+"EY - Sum "+title+".png",dpi=600)
    if show==False:
        plt.close()
    
    minimum=np.min((EY_array[0]-EY_array[1])[peak_start:peak_stop])
    maximum=np.max((EY_array[0]-EY_array[1])[peak_start:peak_stop])
    
    normFactor=np.max((EY_array[0]+EY_array[1])[peak_start:peak_stop])
    
    normMin=minimum/normFactor
    normMax=maximum/normFactor
    
    return (minimum,maximum,normMin,normMax,EY_array,EY_peaks)

def calculate_ratios(energy_array,EY_array,EY_peaks):
    EY_ratio=np.asfarray((EY_array[0][EY_peaks[0]]-EY_array[1][EY_peaks[1]])/(EY_array[0][EY_peaks[0]]+EY_array[1][EY_peaks[1]]))
    
    # EY_ratio=np.asfarray(EY_array[0][EY_peaks[0]]/EY_array[1][EY_peaks[1]])
    EY_ratio=np.abs(EY_ratio)*100
    # print(EY_array[0][EY_peaks[0]],EY_array[1][EY_peaks[1]],EY_ratio)
    return (EY_ratio)

def plot_ratios(ratios,sample,temps,save_path,title,ELY="unknown",element="unknown"):
    fig, ax1 = plt.subplots()    
    ax1.plot(temps,ratios,"o-.",markersize=3,color="navy",label=element+" XMCD")
    ax1.set_title("XMCD - "+element+"$\\ L_{3} \\ Ratio$ - "+ELY,fontsize=FONTSIZE)
    ax1.set_ylabel("L3 % Asymmetry : $\dfrac{|L_{3M_{+}}-L_{3M_{-}}|}{|L_{3M_{+}}+L_{3M_{-}}|}$",fontsize=FONTSIZE-1)
    ax1.set_xlabel("Temperature (K)",fontsize=FONTSIZE-1)
    # ax1.set_ylim(0,2)
    ax1.legend()
    plt.tight_layout()
    plt.savefig(save_path+title+" "+ELY+" - L3 Ratio.png",dpi=600) 
    plt.show()
    
    np.savetxt(save_path+title+" "+ELY+" - L3 Ratio.txt",[temps,ratios])

    return (temps,ratios)

def chunks(l, n):
    """Yield successive n-sized chunks from list l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def peak_locator(series):
    """Finds the location of a peak within a list."""
    peaks=argrelextrema(series, np.greater)
    return peaks

def read(filename): 
    """reads the dat file outputs the list row by row."""
    output=[]
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            output.append(row)
    return (output)

def sort_divide(datfile,size): 
    """from the raw data file, it extracts energy, I0, EY and LY columns."""
    energy=[]        #5th Column
    I0=[]            #7th Column
    EY=[]            #8th Column

    for i in range(size):#from the whole data takes relevant columns 4,6,7,9 --> Energy,I0,EY and LY
        energy.append(datfile[i+11][5])
        I0.append(datfile[i+11][8])
        EY.append(datfile[i+11][9])
           
    energy=np.asfarray(energy)#convert lists to ndarray
    I0=np.asfarray(I0)
    EY=np.asfarray(EY)

    EY=EY/I0

    
    # energy_iter=chunks(energy,size//2) 
    # EY_iter=chunks(EY,size//2)

    
    # energy_array=[]
    # EY_array=[]

    # for i in range(2): #attaches divided array parts
    #     energy_array.append(next(energy_iter))
    #     EY_array.append(next(EY_iter))

        
    # energy_array=np.asfarray(energy_array)
    # EY_array=np.asfarray(EY_array)

    energy_array=np.asfarray(energy)
    EY_array=np.asfarray(EY)
    
    return (energy_array,EY_array)


def average2(a):
    """a,b,c,d are 3x2x1000 dimensional arrays. a[0] energy, a[1] EY, a[2] LY. a[1][0] is EY for horizontal. a[1][1] is EY for vertical"""

    return (a[0],(a[1])/1.0,(a[2])/1.0)   

def average3(a,b,c,d):
    """a,b,c,d are 3x2x1000 dimensional arrays. a[0] energy, a[1] EY, a[2] LY. a[1][0] is EY for horizontal. a[1][1] is EY for vertical"""
    
    energy_array=a[0][0]
    energy_array=np.append([energy_array],[a[0][1]],axis=0)
    
    EY_array=savgol_filter((a[1][0]+b[1][1]+c[1][0]+d[1][1])/4.0,9,2)
    EY_array=np.append([EY_array],savgol_filter([(a[1][1]+b[1][0]+c[1][1]+d[1][0])/4.0],9,2),axis=0)

    return (energy_array,EY_array)   

def average4(data):
    """a,b,c,d are 3x2x1000 dimensional arrays. a[0] energy, a[1] EY, a[2] LY. a[1][0] is EY for horizontal. a[1][1] is EY for vertical"""
    
    data_split=np.split(data,8)
    
    data=np.mean(data_split,axis=0)
    # data=(data_split[5]+data_split[6])/2
    return(data)   

