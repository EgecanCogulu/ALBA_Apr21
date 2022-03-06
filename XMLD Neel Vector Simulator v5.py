# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:41:38 2020

@author: Egecan Cogulu
contact: egecancogulu@gmail.com

Visualization tool for simulating pixel intensity in XMLD-PEEM experiments.

"""
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
from numpy import cross, eye, dot
from scipy.linalg import expm, norm
from matplotlib import cm


global increment #(only) global variable increment controls the increment angle for buttons
increment=5

#Function for calculating the 3D rotation matrix around an "axis" for given angle "theta"
def rotation_Matrix(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))

#Rotation around Z-axis
def Zrotation(X1,Y1,Z1,X2,Y2,Z2,angle,scale=1):
    u=X2-X1
    v=Y2-Y1
    w=Z2-Z1
    
    u2=np.cos(angle)*u-np.sin(angle)*v
    v2=np.sin(angle)*u+np.cos(angle)*v
    w2=w
    return[[X1,Y1,Z1],[X1+u2*scale, Y1+v2*scale, Z1+w2*scale]]

#Rotation around Y-axis
def Yrotation(X1,Y1,Z1,X2,Y2,Z2,angle,scale=1):
    u=X2-X1
    v=Y2-Y1
    w=Z2-Z1
    
    u2=np.cos(angle)*u-np.sin(angle)*w
    v2=v
    w2=-1*np.sin(angle)*u+np.cos(angle)*w
    
   
    return[[X1,Y1,Z1],[X1+u2*scale, Y1+v2*scale, Z1+w2*scale]]

#Rotation function for drawing the arrow heads of the Neel vector (only cosmetic implications)
def custom_rotation(X1,Y1,Z1,X2,Y2,Z2,angle,scale=1):
    
    u=X2-X1
    v=Y2-Y1
    w=Z2-Z1
    my_vector=[u,v,w]
    axis=np.cross(np.cross(my_vector,[0,0,1]),my_vector)
    
    M=rotation_Matrix(axis,angle)
    head=np.dot(M,my_vector)
    u2,v2,w2=head
    return [[X1,Y1,Z1],[X1+u2*scale, Y1+v2*scale, Z1+w2*scale]]

def quiver_data_to_segments(X, Y, Z, u, v, w,length=1):
    arrow = [[X, Y, Z], [X+u*length, Y+v*length, Z+w*length]]
    head1=custom_rotation(X+u*length, Y+v*length, Z+w*length,0,0,0,np.pi/9,0.3) 
    head2=custom_rotation(X+u*length, Y+v*length,Z+w*length,0,0,0,-np.pi/9,0.3)
    segments=[arrow,head1,head2]
    return segments


fig = plt.figure(figsize=(26,7), dpi=None)
ax = fig.add_subplot(1, 2,1, projection='3d')
ax.set_title("Neel Vector Orientation",size=18)
axlimit=1.5
ax.set_xlim3d(-1*axlimit, axlimit)
ax.set_ylim3d(-1*axlimit,axlimit)
ax.set_zlim3d(-1*axlimit,axlimit)

origin=[0,0,0]
coor=[-0.95*axlimit,-0.95*axlimit,-0.95*axlimit]
length=1

qc1="k"
qc2="k"
Q = ax.quiver(*origin,1, 0, 0, color=[qc1,qc2,qc2],  length=1,arrow_length_ratio=0.2)
Q2 = ax.quiver(*origin,-1, 0, 0, color=[qc1,qc2,qc2],  length=1, arrow_length_ratio=0.2)
i = ax.quiver(*coor,0.5, 0, 0, color=['b','b','b'],  length=0.5, arrow_length_ratio=0.2)
j = ax.quiver(*coor,0, 0.5, 0, color=['r','r','r'],  length=0.5, arrow_length_ratio=0.2)
k = ax.quiver(*coor,0, 0, 0.5, color=['g','g','g'],  length=0.5, arrow_length_ratio=0.2)

i2 = ax.quiver(*origin,1, 0, 0, color=['grey','grey','grey'],  length=0.5, arrow_length_ratio=0.1)
j2 = ax.quiver(*origin,0, 1, 0, color=['grey','grey','grey'],  length=0.5, arrow_length_ratio=0.1)
k2 = ax.quiver(*origin,0, 0, 1, color=['grey','grey','grey'],  length=0.5, arrow_length_ratio=0.1)

ax.text(coor[0]+0.35, coor[1], coor[2], "x", color='blue',size=9)
ax.text(coor[0], coor[1]+0.35, coor[2], "y", color='red',size=9)
ax.text(coor[0], coor[1], coor[2]+0.35, "z", color='green',size=9)

# ax.text(0, 1.2, 0, "$\sigma$", color='red',size=9)
# ax.text(0.51,0 , 0.88, "$\pi$", color='blue',size=9)

c1="red"
c2="blue"
ip = ax.quiver(*origin,0, 1, 0, color=[c1,c1,c1],  length=1, arrow_length_ratio=0.1)
oop = ax.quiver(*origin,np.sin(np.pi/6), 0, np.cos(np.pi/6), color=[c2,c2,c2],  length=1, arrow_length_ratio=0.1)


ax.set_xlabel('$X$', fontsize=13)
ax.set_ylabel('$Y$',fontsize=13)
ax.set_zlabel('$Z$', fontsize=13)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
axcolor = 'lightgoldenrodyellow'

axtheta = plt.axes([0.15, 0.045, 0.25, 0.025], facecolor=axcolor)
axphi = plt.axes([0.15, 0.01, 0.25, 0.025], facecolor=axcolor)

resetax = plt.axes([0.15, 0.08, 0.05, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

allignpi = plt.axes([0.205, 0.08, 0.05, 0.04])
button2 = Button(allignpi, 'Allign $\sigma$', color=axcolor, hovercolor='0.975')

allignsigma = plt.axes([0.26, 0.08, 0.05, 0.04])
button3 = Button(allignsigma, 'Allign $\pi$', color=axcolor, hovercolor='0.975')

ipplus = plt.axes([0.315, 0.08, 0.025, 0.04])
button4 = Button(ipplus, '$\phi$ (+)', color="red", hovercolor='lightcoral')

ipminus = plt.axes([0.345, 0.08, 0.025, 0.04])
button5 = Button(ipminus, '$\phi$ (-)',color="red", hovercolor='lightcoral')


oopplus = plt.axes([0.375, 0.08, 0.025, 0.04])
button6 = Button(oopplus, '$\\theta$ (+)', color="blue", hovercolor='lightsteelblue')

oopminus = plt.axes([0.405, 0.08, 0.025, 0.04])
button7 = Button(oopminus, '$\\theta$ (-)', color="blue", hovercolor='lightsteelblue')

flip180 = plt.axes([0.435, 0.08, 0.025, 0.04])
button8 = Button(flip180, '$\\theta$ (180)', color="grey", hovercolor='lightsteelblue')

#drawing the circle in xy plane
theta = np.linspace(0, 2 * np.pi, 201)
x = 1*np.cos(theta)
y = 1*np.sin(theta)
z = 1*np.sin(theta)
ax.plot(x, y, 0,'grey',alpha=0.5)


ax2 = fig.add_subplot(2, 2,2 )
labels = ['$\sigma$ (IP)','$\pi$ (IP & OOP)']
x1,x2=1,1
para = [x1, x2]

x = [-0.002,0.352]  # the label locations
width = 0.2  # the width of the bars


rects = plt.bar(x, para, width,  edgecolor='black',color=("red", "blue"))

ax2.set_ylabel("Component Percentage (%)",size=15)
ax2.set_title('Light Polarization',size=18)

ax2.set_xticks(x)
ax2.set_xlim(-0.12,0.5)
ax2.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
ax2.set_yticklabels([0,"",20,"",40,"",60,"",80,"",100])
ax2.yaxis.set_ticks_position('both')
ax2.tick_params(axis='both', which='major', labelsize=15,labelright=True)
ax2.set_xticklabels(labels)

IMAGE1=np.ones((1,1))*128
ax3 = fig.add_subplot(2, 4,7 )
ax3.tick_params(axis='both',bottom=False,labelbottom=False,left=False,labelleft=False)
ax3.spines['bottom'].set_color('red')
ax3.spines['top'].set_color('red') 
ax3.spines['right'].set_color('red')
ax3.spines['left'].set_color('red')
ax3.set_ylabel("Simulated Pixel Intensity")
ax3.set_xlabel("$\sigma$",size=15)
im1=ax3.imshow(IMAGE1,cmap='gray', vmin=0, vmax=255.)

IMAGE2=np.ones((1,1))*128
ax4 = fig.add_subplot(2, 4,8 )
ax4.tick_params(axis='both',bottom=False,labelbottom=False,left=False,labelleft=False)
im2=ax4.imshow(IMAGE2,cmap='gray', vmin=0, vmax=255.)
ax4.spines['bottom'].set_color('blue')
ax4.spines['top'].set_color('blue') 
ax4.spines['right'].set_color('blue')
ax4.spines['left'].set_color('blue')
ax4.set_ylabel("Simulated Pixel Intensity")
ax4.set_xlabel("$\pi$",size=15)
plt.show()

stheta = Slider(axtheta, '$\phi$ (IP)', 0.001, 360, valinit=180,color="red")
sphi = Slider(axphi, '$\\theta$ (OOP)', 0.001, 180, valinit=90,color="blue")

m1=[[0,0,0],[1,0,0]]

def update(val):
    m1=[[0,0,0],[1,0,0]]
    theta = (np.pi/180)*stheta.val
    phi = (np.pi/180)*sphi.val
    
    # m1=Zrotation(*m1[0],*m1[1],phi)
    # m1=Yrotation(*m1[0],*m1[1],theta)

    segments=quiver_data_to_segments(0,0,0,np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi),np.cos(phi))
    segments2=quiver_data_to_segments(0,0,0,-1*np.cos(theta)*np.sin(phi),-1*np.sin(theta)*np.sin(phi),-1*np.cos(phi))
    Q.set_segments(segments)
    
    Q2.set_segments(segments2)
    sigma_comp=np.abs(np.dot([np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi),np.cos(phi)],[0,1,0]))
    rects[0].set_height(sigma_comp)
    
    # pi_comp=np.abs(np.dot([np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi),np.cos(phi)],[np.sin(np.pi/6), 0, np.cos(np.pi/6)]))
    pi_comp=np.abs(np.dot([np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi),np.cos(phi)],[np.sin(np.pi/6), 0, np.cos(np.pi/6)]))

    rects[1].set_height(pi_comp)
    
    im1.set_array(np.ones((1,1))*256*(1-sigma_comp))
    im2.set_array(np.ones((1,1))*256*(1-pi_comp))
    fig.canvas.draw_idle()

def reset(event):
    sphi.reset()
    stheta.reset()
    
def polpi(event):

    stheta.set_val(90)
    sphi.set_val(90)

def polsigma(event):
    stheta.set_val(360)
    sphi.set_val(30)
    
def ipplus(event):
    stheta.set_val(stheta.val+increment)      

def ipminus(event):
    stheta.set_val(stheta.val-increment)      
    
def flip180(event):
    stheta.set_val(stheta.val+180)     

def oopplus(event):
    sphi.set_val(sphi.val+increment)      

def oopminus(event):
    sphi.set_val(sphi.val-increment)  
    
button.on_clicked(reset)    
    
button2.on_clicked(polpi)
button3.on_clicked(polsigma)

button4.on_clicked(ipplus)
button5.on_clicked(ipminus)

button6.on_clicked(oopplus)
button7.on_clicked(oopminus)

button8.on_clicked(flip180)

stheta.on_changed(update)
sphi.on_changed(update)


