#!/usr/bin/python
# Created by Adam Higgins

import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
from astropy.cosmology import FlatLambdaCDM

__doc__ = """ Produce GRB afterglow spectra and light curves in ISM like
environment from input blastwave parameters following how characteristic
frequencies (nu_c, nu_m) evolve with time using scalings described. For p > 2,
pre-jet break scalings taken from Piran & Narayan, 1998 and post-jet break
scalings taken from Sari, Piran & Halpern, 1999. For p < 2, pre and post-jet
break scalings taken from Dai and Cheng, 2001. This version of the code does
not include self-absorption.
"""


def get_args():
	""" Parse comand line arguments """
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("Model Type",metavar="MODEL",type=str,action="store",
		help="Model type [SPECTRUM/LC]")
	parser.add_argument("Time_0",metavar="T_0",type=float,action="store",
		help="Initial light curve time or spectrum time (s)")
	parser.add_argument("Time_F",metavar="T_f",type=float,action="store",
		help="Final light curve time - not used in spectrum (s)")
	parser.add_argument("Nu_0",metavar="Nu_0",type=float,action="store",
		help="Lowest spectrum frequency or frequency of light curve (Hz)")
	parser.add_argument("Nu_F",metavar="Nu_F",type=float,action="store",
		help="Highest frequency of spectrum - not used for light curve (Hz)")
	parser.add_argument("Jet Ang",metavar="J_Ang",type=float,action="store",
		help="Half-opening angle of the jet (degrees)")
	parser.add_argument("Energy",metavar="E_K",type=float,action="store",
		help="Kinetic energy of the jet (ergs)")
	parser.add_argument("Density",metavar="N",type=float,action="store",
		help="Circum-burst density (cm^-3)")
	parser.add_argument("Epsilon_B",metavar="Eps_B",type=float,action="store",
		help="Fraction of energy with the magnetic field")
	parser.add_argument("Epsilon_E",metavar="Eps_E",type=float,action="store",
		help="Fraction of energy accelerating the electrons")
	parser.add_argument("P",metavar="P",type=float,action="store",
		help="Slope of electron energy distribution")
	parser.add_argument("Redshift",metavar="Z",type=float,action="store",
		help="Redshift")
	parser.add_argument('--Gamma',type=float,default=100,dest='Gamma',
		help='Bulk Lorentz factor (Default = 100)')
	args = parser.parse_args()
	model = args.__dict__['Model Type']
	ti = args.__dict__['Time_0']
	tf = args.__dict__['Time_F']
	nui = args.__dict__['Nu_0']
	nuf = args.__dict__['Nu_F']
	j_ang = args.__dict__['Jet Ang']*np.pi/180
	E_k = args.__dict__['Energy']
	n = args.__dict__['Density']
	eps_b = args.__dict__['Epsilon_B']
	eps_e = args.__dict__['Epsilon_E']
	p = args.__dict__['P']
	z = args.__dict__['Redshift']
	Gamma = args.Gamma
	return model,ti,tf,nui,nuf,j_ang,E_k,n,eps_b,eps_e,p,z,Gamma

	
def bw_props(Gamma,E_k,n,eps_b,p,mp,me,c,e,sigma_t):
	""" Function for calculating blastwave properties from current
	parameters """
	
	# Calculate deceleration timescale
	K = 4
	t_dec = (3*E_k/(4*K**3*np.pi*n*mp*c**5*Gamma**8))**(1/3)
	
	# Calculate physical properties
	B = (32*np.pi*mp*eps_b*n)**(1/2)*Gamma*c    # magnetic field
	P_max = (me*c**2*sigma_t*Gamma*B)/(3*e)    # max power
	R = ((3*E_k)/(4*Gamma**2*n*np.pi*mp*c**2))**(1/3)    # radius
	Ne = (4/3)*np.pi*R**3*n    # number of electrons
	return (t_dec,B,P_max,R,Ne)

	
def spec_flux(flux_max,time,nu,p,nu_m,nu_c):
	""" Function calculates the spectral flux of the afterglow for slow
	and fast cooling regimes at a given frequency from max flux """
	
	# Slow cooling
	if nu_m < nu_c:
		if nu <= nu_m:
			flux_n = flux_max*(nu/nu_m)**(1/3)
		if nu_m < nu <= nu_c:
			flux_n = flux_max*(nu/nu_m)**((1-p)/2)
		if nu_c < nu:
			flux_n = flux_max*(nu_c/nu_m)**((1-p)/2)*(nu/nu_c)**(-p/2)
	
	# Fast cooling			
	if nu_c < nu_m:
		if nu <= nu_c:
			flux_n = flux_max*(nu/nu_c)**(1/3)
		if nu_c < nu <= nu_m:
			flux_n = flux_max*(nu/nu_c)**(-1/2)
		if nu_m < nu:
			flux_n = flux_max*(nu_m/nu_c)**(-1/2)*(nu/nu_m)**(-p/2)
	return flux_n

	
def model_flux(t_dec,B,P_max,R,Ne,d_l,z,mp,me,e,c,sigma_t,time,nu,Gamma,E_k,
	n,eps_b,eps_e,p,j_ang):
	""" Function for deriving the flux for the spectrum or light curve at
	given times and frequencies """
	
	# calculate lorentz factors, characteristic frequencies and
	# jet break time
	gamma_m = Gamma*eps_e*((p-2)/(p-1))*(mp/me)   
	gamma_c = (6*np.pi*me*c)/(sigma_t*Gamma*B**2*time)
	gamma_crit = (6*np.pi*me*c)/(sigma_t*Gamma*B**2*t_dec)
	t_jb = 86400*(((1/0.057)*j_ang*((1+z)/2)**(3/8)*(E_k/1e53)**(1/8)*
		(n/0.1)**(-1/8))**(8/3))
	nu_m0 = (gamma_m**2*Gamma*e*B)/(2*np.pi*me*c)
	nu_c0 = (gamma_c**2*Gamma*e*B)/(2*np.pi*me*c)
	flux_max = (Ne*P_max*1e26)/(4*np.pi*d_l**2)
	
	# At times smaller than the deceleration timescale
	if time <= t_dec:
		flux_n = spec_flux(flux_max,time,nu,p,nu_m0,nu_c0)
		flux_n = flux_n*(time/t_dec)**3
		return flux_n
	
	# At times greater than the deceleration timescale	
	if time > t_dec:	
		if p > 2:
			nu_m = nu_m0*(time/t_dec)**(-3/2)
			nu_c = nu_c0*(time/t_dec)**(-1/2)	
		
		if p < 2:
			nu_m = nu_m0*(time/t_dec)**((-3*(p+2))/(8*(p-1)))
			nu_c = nu_c0*(time/t_dec)**(-1/2)
		
		if time > t_jb:
			nu_c = nu_c0*(t_jb/t_dec)**(-1/2)	
			flux_max = flux_max*(time/t_jb)**(-1)
			
			if p > 2:
				nu_m = nu_m0*(t_jb/t_dec)**(-3/2)*(time/t_jb)**(-2)
				
			if p < 2:
				nu_m = (nu_m0*(t_jb/t_dec)**((-3*(p+2))/(8*(p-1)))*(time/t_jb)
					**(-(p+2)/(2*(p-1))))
			
		flux_n = spec_flux(flux_max,time,nu,p,nu_m,nu_c)
		return flux_n

		
def ag_mods(model,ti,tf,nui,nuf,j_ang,E_k,n,eps_b,eps_e,p,z,Gamma):
	""" Creates GRB afterglow light curves or spectra """

	# define physical constants in cgs units
	c = 2.998e10
	me = 9.109e-28
	mp = 1.673e-24
	e = 4.803e-10
	sigma_t = 6.652e-25

	# Calculate luminosity distance
	cosmo = FlatLambdaCDM(H0=70,Tcmb0=2.725,Om0=0.3)
	d_l = (cosmo.luminosity_distance(z).value*3.086e+24)

	t_dec,B,P_max,R,Ne = bw_props(Gamma,E_k,n,eps_b,p,mp,me,c,e,sigma_t)

	if model == 'SPECTRUM':
		freq = np.logspace(np.log10(nui),np.log10(nuf),1000)
		ymod = np.zeros(len(freq))
		for i in range(len(freq)):
			ymod[i] = model_flux(t_dec,B,P_max,R,Ne,d_l,z,mp,me,e,c,sigma_t,
				ti,freq[i],Gamma,E_k,n,eps_b,eps_e,p,j_ang)
					  
		plt.figure()
		plt.plot(freq,ymod,color='black',marker=' ')
		plt.xlabel('$\\nu$ (Hz)')
		plt.ylabel('Flux Density (mJy)')
		plt.xscale('log')
		plt.yscale('log')
		#plt.savefig('spectrum_mod.png')
		plt.show()
		
	if model == 'LC':
		times = np.logspace(np.log10(ti),np.log10(tf),1000)
		ymod = np.zeros(len(times))
		for i in range(len(times)):
			ymod[i] = model_flux(t_dec,B,P_max,R,Ne,d_l,z,mp,me,e,c,sigma_t,
				times[i],nui,Gamma,E_k,n,eps_b,eps_e,p,j_ang)		

		plt.figure()
		plt.plot(times,ymod,color='black',marker=' ')
		plt.xlabel('Time since trigger (s)')
		plt.ylabel('Flux Density (mJy)')
		plt.xscale('log')
		plt.yscale('log')
		#plt.savefig('lightcurve_mod.png')
		plt.show()
	return 0
	

def main():
	""" Run script from command line """
	model,ti,tf,nui,nuf,j_ang,E_k,n,eps_b,eps_e,p,z,Gamma = get_args()
	
	# Remove unphysical parameters	
	if (p <= 1 or p == 2):
		print("Electron distribution slope, p only valid above 1")
		sys.exit()		
	return ag_mods(model,ti,tf,nui,nuf,j_ang,E_k,n,eps_b,eps_e,p,z,Gamma)


if __name__ == '__main__':
    sys.exit(main())