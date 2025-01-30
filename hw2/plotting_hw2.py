import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
def plots(E,phi):
    plt.figure()
    plt.title(r'$\phi(E)$')
    plt.plot(E, phi,label = r'$\phi(E)$')
    plt.xscale('log')
    plt.xlabel('Energy (eV)')
    plt .ylabel(r'$\phi$')
    plt.grid(True, which = 'both')
    plt .legend()
    plt.savefig("charts/561hw2q1_phiU.png")
    plt.close()

    #plot 2
    plt.plot(E[62144:80588], phi[62144:80588],label = r'$\phi(E)$')
    plt.title('1 - 40 eV')
    plt.xlabel('Energy (eV)')
    plt.ylabel(r'$\phi$')
    plt.grid(True, which='both')
    plt.savefig("charts/561hw2q1_40eV.png")
    plt.close()
    
    #plot 3
    plt.plot(E[34062:34538], phi[34062:34538],label = r'$\phi(E)$')
    plt.title('10 - 11 keV')
    plt.xlabel('Energy (eV)')
    plt.ylabel(r'$\phi$')
    plt.grid(True, which='both')
    plt.savefig("charts/561hw2q1_10keV.png")
    plt.close()

