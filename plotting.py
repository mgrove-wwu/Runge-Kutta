import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def PhaseSpace2D(x, drop, ex, lb):
    
    #forget the transient state at the beginning:
    x = x[int(len(x)*drop):]
    
    #plot phase space:
    plt.figure('sheet 4, exercise 1'+ex,figsize=(5.8, 4.1))
    plt.xlabel(r'$y$',fontsize=18)
    plt.ylabel(r'$dy$',fontsize=18)

    plt.plot(x[:,0],x[:,1],'b',linewidth=2,label=lb)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('ex1{}.pdf'.format(ex))

def PhaseSpace3D(t, x, drop, ex, lb):
    x = x[int(len(x)*drop):]

    fig = plt.figure('sheet 4, exercise 2'+ex,figsize=(5.8, 4.1))
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'$x$',fontsize=18)
    ax.set_ylabel(r'$y$',fontsize=18)
    ax.set_zlabel(r'$z$',fontsize=18) 

    ax.plot(x[:,0],x[:,1],x[:,2],'g',linewidth=2,label=lb)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('ex1{}.pdf'.format(ex))

def LorenzMap(t, x, drop, ex, lb):
    phi = x[:,0]
    max_phi = phi[(phi > np.roll(phi,1)) & (phi > np.roll(phi,-1))]
    max_phi = max_phi[1:-1] #first and last element of max_phi can be misleading
    #if phi(T)<phi(t0)>phi(t0+h) and phi(T-h)<phi(T)>phi(t0), respectively

    plt.figure(r'sheet 4, exercise 1'+ex+', return map',figsize=(5.8, 4.1))
    plt.xlabel(r'$\mathrm{max}_{n}$',fontsize=18)
    plt.ylabel(r'$\mathrm{max}_{n+1}$',fontsize=18)

    plt.plot(max_phi[:-1], max_phi[1:], 'bo',label=lb)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('ex1{}_return_map.pdf'.format(ex))

def SolutionPlot(t, x, drop, ex, cl, lb):
    #forget the transient state at the beginning:
    t = t[int(len(t)*drop):]
    x = x[int(len(x)*drop):]

    #plot phase space:
    plt.figure(r'sheet 6 '+ex,figsize=(5.8, 4.1))
    plt.xlabel(r'$t$',fontsize=18)
    plt.ylabel(r'$y$',fontsize=18)

    plt.plot(t,x,cl,linewidth=2,label=lb)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('{}.pdf'.format(ex))
