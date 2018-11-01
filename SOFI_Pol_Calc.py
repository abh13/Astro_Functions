#!/usr/bin/python
# -*- coding: utf-8 -*-
# SOFI polarisation calibration script v1.0
# Created by Adam Higgins, Stefano Covino and Klaas Wiersema
# Email: abh13@le.ac.uk, kw113@le.ac.uk

__doc__ = """ Calibration script used for reducing optical linear polarimetric
observations made using the SOFI instrument on-board the NTT at La Silla.
The prescription for the ordinary and extraordinary beams is extraordinary
for the top beam, ordinary for the bottom - opposite to our EFOSC2 work. This
applies to the standard star measurements too!
Please make sure the eight input files follow the naming
convention - angleXXX_ord.1/angleXXX_exord.1 """

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import emcee
from scipy.interpolate import interp1d
import corner
import argparse

def get_args():
	# Parse command line arguments
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("Directory",metavar="DIR",type=str,action="store",
		help="Required directory")
	parser.add_argument("Filter",metavar="FILTER",type=str,action="store",
		help="Observed filter - choices are Z")
	parser.add_argument("Par Angle",metavar="PAR",type=float,action="store",
		help="Parallactic Angle (deg)")
	parser.add_argument("Gain",metavar="GAIN",type=float,action="store",
		help="Input Gain")
		
	args = parser.parse_args()
	folder_path = args.__dict__['Directory']
	wave_band = args.__dict__['Filter']
	par_ang = args.__dict__['Par Angle']
	gain = args.__dict__['Gain']
	return folder_path,gain,wave_band,par_ang
    
def sofi_cal_mm(folder_path,standard_star_file,mirror_props_file,waveband,
par_ang,q_values,u_values):
	""" Mueller Matrix Method """
	
	# Unpolarised standard star data
	data = np.genfromtxt(standard_star_file,delimiter=',',dtype=float,
		usecols=(1,2,3,4,5))
	un_par_angle = data[:,0]
	un_q = data[:,1]
	un_qerr = data[:,2]
	un_u = data[:,3]
	un_uerr = data[:,4]

	# Refractive index properties of the aluminium mirror
	data = np.genfromtxt(mirror_props_file,delimiter = ',',skip_header=1)
	mask = np.ma.masked_outside(data[:,0],0.2,2.9)
	lambdam = data[:,0][mask.mask == False]
	nm = data[:,1][mask.mask == False]
	km = data[:,2][mask.mask == False]
	ni = interp1d(lambdam, nm, kind='linear', bounds_error=False)
	ki = interp1d(lambdam, km, kind='linear', bounds_error=False)

	def MirrorMatrixComplex(n,k,theta):
		# Mueller matrix for m3 mirror
		p = n**2 - k**2 - np.sin(theta)**2
		q = 2*n*k
		s = np.sin(theta)*np.tan(theta)
		r_pos = np.sqrt(p + np.sqrt(p**2 + q**2))/np.sqrt(2)
		r_neg = np.sqrt(-p + np.sqrt(p**2 + q**2))/np.sqrt(2)
		rho = np.sqrt((np.sqrt(p**2 + q**2) + s**2 - 2*s*r_pos)/
			(np.sqrt(p**2 + q**2) + s**2 + 2*s*r_pos))
		delta = np.arctan2(2*s*r_neg,np.sqrt(p**2 + q**2) - s**2)
		row1 = [1 + rho**2, 1 - rho**2, 0, 0]
		row2 = [1 - rho**2, 1 + rho**2, 0, 0]
		row3 = [0, 0, -2*rho*np.cos(delta), -2*rho*np.sin(delta)]
		row4 = [0, 0, 2*rho*np.sin(delta), -2*rho*np.cos(delta)]
		return 0.5*np.matrix([row1, row2, row3, row4])

	def RotationMatrix(phi):
		# Mueller rotation matrix accounts for different orientations of the
		# mirror
		row1 = [1, 0, 0, 0]
		row2 = [0, np.cos(2*phi), np.sin(2*phi), 0]
		row3 = [0, -np.sin(2*phi), np.cos(2*phi), 0]
		row4 = [0, 0, 0, 1]
		return np.matrix([row1, row2, row3, row4])

	def SOFI_Matrix(par_angle,n,k,offset):
		# Final matrix for SOFI instrument
		m_rot1 = RotationMatrix(np.radians(offset))
		m_rot2 = RotationMatrix(np.radians(-1*(par_angle)))
		m_m3 = MirrorMatrixComplex(n,k,np.radians(45)) #M3 mirror
		m_rot3 = RotationMatrix(np.radians(-1*(par_angle)))
		m_mirror = MirrorMatrixComplex(n,k,np.radians(0))
		return m_rot1*m_rot2*m_m3*m_rot3*m_mirror

	def instrument_pol_function(waveband,q,u,par_angle,f,offset):
		# Function multiples the SOFI matrix with the measured stoke
		# parameters
		n = ni(waveband)
		k = ki(waveband)
		stokes = np.matrix([1, q, u, 0]).transpose()
		s = SOFI_Matrix(par_angle,f*n,f*k,offset)*stokes
		return s

	def lnprior(parms):
		# Some physical priors
		f, offset = parms		
		if 0 < f < 3 and -180 < offset < 180:
			return 0.0			
		return -np.inf

	def lnprob(parms,waveband,par_angle,q,u,q_err,u_err):
		# Posterior probability to find the angle offset and multiplication
		# factor for mirror. Uses maximum log-likelihood method.
		lp = lnprior(parms)
		
		if not np.isfinite(lp):
			return -np.inf 
			
		f, offset = parms
		q_mod = np.zeros(len(q))
		u_mod = np.zeros(len(q))
		
		for i in range(len(q)):
			s = instrument_pol_function(waveband,0,0,par_angle[i],f,offset)
			q_mod[i] = s[1,0]/s[0,0]
			u_mod[i] = s[2,0]/s[0,0]
			
		q_ls = -0.5*(np.sum((((q - q_mod)/q_err)**2) + np.log(q_err**2))
			+ len(q)*np.log(2*np.pi))
		u_ls = -0.5*(np.sum((((u - u_mod)/u_err)**2) + np.log(u_err**2))
			+ len(u)*np.log(2*np.pi))
		result = lp + q_ls + u_ls
		return result

	# Set up initial values and run MCMC 
	nwalkers, ndim, nsteps = 20, 2, 2500
	guess = np.zeros([nwalkers,ndim])
	
	for i in range(nwalkers):
		angle_0 = np.random.uniform(-120,60)
		f_0 = np.random.uniform(0.5,2.0)
		guess[i] = [f_0,angle_0]
		
	sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(waveband,
		un_par_angle,un_q,un_u,un_qerr,un_uerr))
	print("Running MCMC (~2-3 mins)")
	sampler.run_mcmc(guess,nsteps)

	# Samples including burn-in and with modified reflective boundaries
	samples = sampler.chain
	samples[samples > 60] = samples[samples > 60] - 180
	samples[samples < -120] = samples[samples < -120] + 180

	# Gelman-Rubin test for chain convergance
	def gelman_rubin(chain):
		ssq = np.var(chain, axis=1, ddof=1)
		w = np.mean(ssq, axis=0)
		theta_b = np.mean(chain, axis=1)
		theta_bb = np.mean(theta_b, axis=0)
		m = float(chain.shape[0])
		n = float(chain.shape[1])
		B = n / (m - 1.0) * np.sum((theta_bb - theta_b)**2, axis=0)
		var_theta = (n - 1.0) / n * w + 1.0 / n * B
		statistic = np.sqrt(var_theta / w)
		return statistic

	mf_gr, do_gr = gelman_rubin(samples)
	mf_conv = []
	do_conv = []
	
	if mf_gr < 1.1:
		mf_conv = "- chains converged!"
		
	else:
		mf_conv = "- chains did not converge!"
		
	if do_gr < 1.1:
		do_conv = "- chains converged!"
		
	else:
		do_conv = "- chains did not converge!"
		
	# Clean samples after removing burn-in with modified reflective	boundaries
	samples_clean = sampler.chain[:,250:,:].reshape((-1,ndim))
	samples_clean[samples_clean<60] = samples_clean[samples_clean<60] + 180
	samples_clean[samples_clean>-120] = samples_clean[samples_clean>-120]-180
	p1, p2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(
		samples_clean, [16, 50, 84],axis=0)))

	print("Refractive index Multiplication factor: {0} (+{1} -{2})"
		.format(round(p1[0],3),round(p1[1],3),round(p1[2],3)))
	print("Gelman-Rubin Statistic:",mf_gr,mf_conv)
	print("\nDetector offset (degrees): {0} (+{1} -{2})".
		format(round(p2[0],2),round(p2[1],2),round(p2[2],2)))
	print("Gelman-Rubin Statistic:",do_gr,do_conv)
	print('')
	f_est = p1[0]
	ferr_est = (p1[1]+p1[2])/2
	offset_est = p2[0]
	offseterr_est = (p2[1]+p2[2])/2

	# Print out numerical values for the two mirror matrices
	print('Mirror Matrix (0 degrees):')
	print(MirrorMatrixComplex(ni(waveband),ki(waveband),np.radians(0)))
	print('\nMirror Matrix (45 degrees):')
	print(MirrorMatrixComplex(ni(waveband),ki(waveband),np.radians(45)))
	print('')

	# Plot and save the model for the instrumental polarisation
	xx = np.linspace(-180,180,360)
	qmod = []
	umod = []
	
	for i in range(len(xx)):
		un_instmod = instrument_pol_function(waveband,0,0,xx[i],
			f_est,offset_est)
		qmod.append(un_instmod[1,0]/un_instmod[0,0])
		umod.append(un_instmod[2,0]/un_instmod[0,0])

	plt.figure()
	one = plt.errorbar(un_par_angle,un_q,yerr=un_qerr,color='red',
		fmt='.',label='Q')
	two = plt.errorbar(un_par_angle,un_u,yerr=un_uerr,color='blue',
		fmt='.',label='U')
	plt.plot(xx,qmod,color='black')
	plt.plot(xx,umod,color='black')
	plt.xlabel('Parallactic Angle ($^{\circ}$)')
	plt.xlim(-180,180)
	plt.legend(handles=[one,two])
	modelfile = folder_path + 'best_model.png'
	plt.savefig(modelfile)

	# Plot and save corner plot
	fig = corner.corner(samples_clean, labels=["MF","$\phi_{offset}$"])
	cornerfile = folder_path + 'pol_param_corner.png'
	fig.savefig(cornerfile)

	# Plot and save walker paths
	fig, ax = plt.subplots(2,1,sharex=True)
	ax[0].plot(samples[:, :, 0].T, color='grey')
	ax[0].axhline(f_est, color='red')
	ax[0].set_ylabel("MF")
	ax[1].plot(samples[:, :, 1].T, color='grey')
	ax[1].axhline(offset_est, color='red')
	ax[1].set_ylabel("$\phi_{offset}$")
	pathfile = folder_path + 'walker_paths.png'
	fig.savefig(pathfile)

	# Calculate real polarisations from raw measured values using the SOFI
	# matrix and estimated offset and Mf parameters calculated above
	real_q = []
	real_u = []
	
	for i in range(len(q_values)):
		stokes = np.matrix([1,q_values[i],u_values[i],0]).transpose()
		Emat = SOFI_Matrix(par_ang,f_est*ni(waveband),f_est*ki(waveband),offset_est)
		result = Emat.I*stokes
		real_q.append(result[1,0]/result[0,0])
		real_u.append(result[2,0]/result[0,0])
		
	return real_q,real_u

def sofi_cal_sa(folder_path,standard_star_file,mirror_props_file,waveband,
par_ang,q_values,u_values):
	""" Semi-analytical method """

	data = np.genfromtxt(standard_star_file,delimiter=',')
	sourcenames = data[:,0]
	x_data = data[:,1]
	y1 = data[:,2]
	y1_err = data[:,3]
	y2 = data[:,4]
	y2_err = data[:,5]
	xx = np.linspace(-180,180,200)

	def lnprior(parms):
		# Set up priors using physical limits
		p1, p2 = parms
		if 0 < p1 < 0.1 and 0 < p2 < 180:
			return 0.0
		return -np.inf

	def logprob(parms, x, y1, y2, y1_err, y2_err):
		# Define the posterior
		lp = lnprior(parms)
		
		if not np.isfinite(lp):
			return -np.inf
			
		p1, p2 = parms
		y1_ls = -0.5*(np.sum((((y1-p1*np.cos(np.deg2rad((2*x+p2))))/y1_err)
			**2) + np.log(y1_err**2) + len(y1)*np.log(2*np.pi)))
		y2_ls = -0.5*(np.sum((((y2-p1*np.cos(np.deg2rad((2*x+p2-90))))/y2_err)
			**2) + np.log(y2_err**2)) + len(y2)*np.log(2*np.pi))
		result = y1_ls + y2_ls
		return result

	# Find the best fit
	nwalkers, ndim, nsteps = 20, 2, 10000
	guess = [0.05,90]
	p0 = [guess+(0.002*np.random.randn(ndim)) for i in range(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers,ndim,logprob,
		args=(x_data,y1,y2,y1_err,y2_err))
	print("Running MCMC... (~ 2-3 mins)\n")
	sampler.run_mcmc(p0,nsteps)
	samples = sampler.chain
	samples_r = sampler.chain[:,1000:,:].reshape((-1,ndim))

	# Print the parameters and uncertainties at 1 sigma
	p1, p2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
		zip(*np.percentile(samples_r, [16, 50, 84],axis=0)))
	params = [p1[0],p2[0]]

	print('SOFI PQ Instrumental Polarisation: ')
	print("""{0}(±{1})*cos(2x + {2}(±{3}))\n""".format(round(p1[0],3),
		round((p1[1]+p1[2])/2,4),round(p2[0],3),round((p2[1]+p2[2])/2,3)))

	print('SOFI PU Instrumental Polarisation: ')
	print("""{0}(±{1})*cos(2x + {2}(±{3}))""".format(round(p1[0],3),round(
		(p1[1]+p1[2])/2,4),round((p2[0]-90),3),round((p2[1]+p2[2])/2,3)))

	# Plot data and models
	def mod1(x,p1,p2):
		return p1*np.cos(np.deg2rad((2*x+p2)))

	def mod2(x,p1,p2):
		return p1*np.cos(np.deg2rad((2*x+p2-90)))

	y1_mod = mod1(xx,*params)
	y2_mod = mod2(xx,*params)

	plt.figure()
	one = plt.errorbar(x_data,y1,yerr=y1_err,color='red',fmt='.',label='Q')
	two = plt.errorbar(x_data,y2,yerr=y2_err,color='blue',fmt='.',label='U')
	plt.plot(xx,y1_mod,color='black')
	plt.plot(xx,y2_mod,color='black')
	plt.xlabel('Parallactic Angle ($^{\circ}$)')
	plt.xlim(-180,180)
	plt.legend(handles=[one,two])
	modelfile = folder_path + 'best_model.png'
	plt.savefig(modelfile)

	# Plot and save corner plot
	fig = corner.corner(samples_r, labels=["Wave Amp","$\phi$"])
	cornerfile = folder_path + 'pol_param_corner.png'
	fig.savefig(cornerfile)

	# Plot and save walker paths
	fig, ax = plt.subplots(2,1,sharex=True)
	ax[0].plot(samples[:, :, 0].T, color='grey')
	ax[0].axhline(p1[0], color='red')
	ax[0].set_ylabel("Wave Amp")
	ax[1].plot(samples[:, :, 1].T, color='grey')
	ax[1].axhline(p2[0], color='red')
	ax[1].set_ylabel("$\phi$")
	pathfile = folder_path + 'walker_paths.png'
	fig.savefig(pathfile)

	# Calculate real q and u
	real_q = []
	real_u = []
	
	for i in range(len(q_values)):
		real_q.append(q_values[i] - mod1(par_ang,*params))
		real_u.append(u_values[i] - mod2(par_ang,*params))
		
	return real_q,real_u

def sofi_pol(folder_path,gain,wave_band,par_ang):
	""" This script calibrates the instrumental polarisation of the
	SOFI instrument on board the NTT (using the functions above) and 
	calculates the polarisation of the sources. """

	# Load relevant files
	file_name_ord0 = 'angle0_ord.1'
	file_name_ord45 = 'angle45_ord.1'
	file_name_ord90 = 'angle90_ord.1'
	file_name_ord135 = 'angle135_ord.1'
	file_name_ext0 = 'angle0_exord.1'
	file_name_ext45 = 'angle45_exord.1'
	file_name_ext90 = 'angle90_exord.1'
	file_name_ext135 = 'angle135_exord.1'
		
	ordin_0 = os.path.join(folder_path,file_name_ord0)
	ordin_45 = os.path.join(folder_path,file_name_ord45)
	ordin_90 = os.path.join(folder_path,file_name_ord90)
	ordin_135 = os.path.join(folder_path,file_name_ord135)
	extra_0 = os.path.join(folder_path,file_name_ext0)
	extra_45 = os.path.join(folder_path,file_name_ext45)
	extra_90 = os.path.join(folder_path,file_name_ext90)
	extra_135 = os.path.join(folder_path,file_name_ext135)

	# Defines two lists of ordinary and extra ordinary files to extract data
	ordinary_beam = [ordin_0,ordin_45,ordin_90,ordin_135]
	extra_beam = [extra_0,extra_45,extra_90,extra_135]

	def beam_data(beam_angle):
		# Extracts data for all targets per angle of selected beam
		total_data = {}
		target_list = []
		
		f = open(beam_angle,'r')
		data = np.genfromtxt(beam_angle,delimiter='',dtype=float,
			skip_header=1,usecols=[1,2,3,4,5,6,7],unpack=True)
		x_centre = np.atleast_1d(data[0])
		y_centre = np.atleast_1d(data[1])
		fluxbgs = np.atleast_1d(data[2])
		sourcearea = np.atleast_1d(data[3])
		meanbg = np.atleast_1d(data[4])
		bgstd = np.atleast_1d(data[5])
		bgarea = np.atleast_1d(data[6])
		
		for i in range(len(x_centre)):
			name = 'Source '+ str(i+1)
			total_data[name] = {'x': x_centre[i], 'y': y_centre[i],
				'flux': fluxbgs[i], 'area': sourcearea[i],'msky': meanbg[i],
				'st dev': bgstd[i], 'n sky': bgarea[i]}
				
		f.close()
		return total_data

	def target_list(data_list):
		# Creates a target list with target and field stars
		target_list = []

		for i in range(0,len(data_list),1):
			name = 'Source '+ str(i+1)
			target_list.append(name)

		return target_list

	# Data from file for each angle of each beam stored in these dictionaries
	# and target list stored in an array. Raise error if files or folder cannot be
	# found!!
	try:
		ordin_data_0 = beam_data(ordinary_beam[0])
		ordin_data_45 = beam_data(ordinary_beam[1])
		ordin_data_90 = beam_data(ordinary_beam[2])
		ordin_data_135 = beam_data(ordinary_beam[3])
		extra_data_0 = beam_data(extra_beam[0])
		extra_data_45 = beam_data(extra_beam[1])
		extra_data_90 = beam_data(extra_beam[2])
		extra_data_135 = beam_data(extra_beam[3])

	except FileNotFoundError as e:
		print('Cannot find the folder or files! Please check input!!')
		sys.exit()
		
	target_list = target_list(ordin_data_0)

	# Ensure all angles in both ordinary and extraordinary beams have the
	# same number of sources
	if (len(ordin_data_0) or len(ordin_data_45) or len(ordin_data_90) or
	len(ordin_data_135)) != (len(extra_data_0) != len(extra_data_45) or
	len(extra_data_90) or len(extra_data_135)):
		print('One or more data files have unequal numbers of sources!')
		sys.exit()

	def flux_error(beam_info,target_list):
		# Calculates flux uncertainty for each source per angle of each	beam
		flux_error = []
		k = 1
		nd = 1
		eta = 1
	   
		for i in range(0,len(target_list),1):
			flux_err1 = beam_info[target_list[i]]['flux']/(gain*eta*nd)
			flux_err2 = (beam_info[target_list[i]]['area']*
				beam_info[target_list[i]]['st dev']*
				beam_info[target_list[i]]['st dev'])
			flux_err3 = ((k/beam_info[target_list[i]]['n sky'])*
				(beam_info[target_list[i]]['area']*
				beam_info[target_list[i]]['st dev'])**2)
			flux_error_calc = np.sqrt(flux_err1 + flux_err2 + flux_err3)
			flux_error.append(flux_error_calc)
			
		return flux_error

	# Flux errors stored in following arrays
	ordin_fluxerr_0 = flux_error(ordin_data_0,target_list)
	ordin_fluxerr_45 = flux_error(ordin_data_45,target_list)
	ordin_fluxerr_90 = flux_error(ordin_data_90,target_list)
	ordin_fluxerr_135 = flux_error(ordin_data_135,target_list)

	extra_fluxerr_0 = flux_error(extra_data_0,target_list)
	extra_fluxerr_45 = flux_error(extra_data_45,target_list)
	extra_fluxerr_90 = flux_error(extra_data_90,target_list)
	extra_fluxerr_135 = flux_error(extra_data_135,target_list)

	def norm_flux(ordin_beam,extra_beam,ordin_fluxerr,extra_fluxerr,
	target_list):
		# Calculates the normalised flux per angle for each beam and the error
		# on the flux
		norm_flux_value = []
		norm_flux_err = []
	   
		for i in range(0,len(target_list),1):
			nf1 = (ordin_beam[target_list[i]]['flux']-
				extra_beam[target_list[i]]['flux'])
			nf2 = (ordin_beam[target_list[i]]['flux']+
				extra_beam[target_list[i]]['flux'])
			norm_flux = nf1/nf2
			norm_flux_value.append(norm_flux)
			nfe1 = np.sqrt((ordin_fluxerr[i]**2)+(extra_fluxerr[i]**2))
			nfe2 = (ordin_beam[target_list[i]]['flux']-
				extra_beam[target_list[i]]['flux'])
			nfe3 = np.sqrt((ordin_fluxerr[i]**2)+(extra_fluxerr[i]**2))
			nfe4 = (ordin_beam[target_list[i]]['flux']+
				extra_beam[target_list[i]]['flux'])
			norm_flux_error = norm_flux*np.sqrt((nfe1/nfe2)**2+(nfe3/nfe4)**2)
			norm_flux_err.append(norm_flux_error)
			
		return(norm_flux_value,norm_flux_err)

	# Normalised flux values and errors stored in following arrays
	norm_flux_0,norm_flux_err_0 = norm_flux(ordin_data_0,extra_data_0,
		ordin_fluxerr_0,extra_fluxerr_0,target_list)
	norm_flux_45,norm_flux_err_45 = norm_flux(ordin_data_45,extra_data_45,
		ordin_fluxerr_45,extra_fluxerr_45,target_list)
	norm_flux_90,norm_flux_err_90 = norm_flux(ordin_data_90,extra_data_90,
		ordin_fluxerr_90,extra_fluxerr_90,target_list)
	norm_flux_135,norm_flux_err_135 = norm_flux(ordin_data_135,extra_data_135,
		ordin_fluxerr_135,extra_fluxerr_135,target_list)

	# Coversion factor from degrees to radians
	dtr = np.pi/180

	def pol_parameters(norm_flux_0,norm_flux_45,norm_flux_90,norm_flux_135,
	target_list):
		# Calculates the measured Q, U, and P values of the objects
		q_values = []
		u_values = []
		p_values = []
		
		for i in range(0,len(target_list),1):
			q = ((0.5*norm_flux_0[i]*np.cos(4*0*dtr))+(0.5*norm_flux_45[i]*
				np.cos(4*22.5*dtr))+(0.5*norm_flux_90[i]*np.cos(4*45*dtr))+
				(0.5*norm_flux_135[i]*np.cos(4*67.5*dtr)))
			u = ((0.5*norm_flux_0[i]*np.sin(4*0*dtr))+(0.5*norm_flux_45[i]*
				np.sin(4*22.5*dtr))+(0.5*norm_flux_90[i]*np.sin(4*45*dtr))+
				(0.5*norm_flux_135[i]*np.sin(4*67.5*dtr)))
			p = np.sqrt((q**2)+(u**2))
			q_values.append(q)
			u_values.append(u)
			p_values.append(p)	
			
		return(q_values,u_values,p_values)

	# Q, U, and P values in following arrays
	q_values,u_values,p_values = pol_parameters(norm_flux_0,norm_flux_45,
		norm_flux_90,norm_flux_135,target_list)
	
	# Account for SOFI instrumental polarisation. If the waveband isn't Z,
	# then the program terminates
	mirror_props_file = 'METALS_Aluminium_Rakic.txt'
	print('')	
	
	if wave_band == 'Z':
		real_q,real_u = sofi_cal_mm(folder_path,'pz_standards.txt',
			mirror_props_file,0.90,par_ang,q_values,u_values)
			
	else:
		print('Code does not calibrate for this filter! Please check input!!')
		sys.exit()

	# Calculate real value of p
	real_p = []
	
	for i in range(0,len(target_list),1):
		real_p.append(np.sqrt((real_q[i]**2)+(real_u[i]**2)))
	   
	def calc_theta(real_u,real_q):
		# Calculates theta for all objects
		theta_values = []
		
		for i in range(0,len(target_list),1):
			theta_values.append(0.5*np.arctan(real_u[i]/real_q[i])*(1/dtr))
			
		return theta_values

	def position_angle(theta_values,real_q,real_u,target_list):
		# Calculate proper position angles
		corr_theta_values = []
		
		for i in range(0,len(target_list),1):
			if real_q[i] < 0:
				corr_theta_values.append(theta_values[i]+90)
			if real_q[i] > 0 and real_u[i] > 0:
				corr_theta_values.append(theta_values[i]+0)
			if real_q[i] > 0 and real_u[i] < 0:
				corr_theta_values.append(theta_values[i]+180)
				
		return corr_theta_values

	# Theta value arrays
	theta_values = calc_theta(real_u,real_q)
	corr_theta_values = position_angle(theta_values,real_q,real_u,target_list)

	def parameter_errors(norm_flux_err_0,norm_flux_err_45,norm_flux_err_90,
	norm_flux_err_135,real_p,ordin_data_0,extra_data_0,ordin_data_45,
	extra_data_45,ordin_data_90,extra_data_90,ordin_data_135,extra_data_135,
	target_list):
		# Calculate errors on Q, U, P, Theta and the SD of the average flux
		q_errors = []
		u_errors = []
		sig_p = []
		flux_sig = []
		theta_errors = []

		for i in range(0,len(target_list),1):
			q_errors.append(np.sqrt(((0.5*norm_flux_err_0[i]*np.cos(0*dtr))
				**2)+((0.5*norm_flux_err_45[i]*np.cos(22.5*dtr))**2)+((0.5*
				norm_flux_err_90[i]*np.cos(45*dtr))**2)+((0.5*
				norm_flux_err_135[i]*np.cos(67.5*dtr))**2)))
			u_errors.append(np.sqrt(((0.5*norm_flux_err_0[i]*np.sin(0*dtr))
				**2)+((0.5*norm_flux_err_45[i]*np.sin(22.5*dtr))**2)+((0.5*
				norm_flux_err_90[i]*np.sin(45*dtr))**2)+((0.5*
				norm_flux_err_135[i]*np.sin(67.5*dtr))**2)))
			flux_sig.append(1/(np.sqrt((ordin_data_0[target_list[i]]['flux']+
				extra_data_0[target_list[i]]['flux']+
				ordin_data_45[target_list[i]]['flux']+
				extra_data_45[target_list[i]]['flux']+
				ordin_data_90[target_list[i]]['flux']+
				extra_data_90[target_list[i]]['flux']+
				ordin_data_135[target_list[i]]['flux']+
				extra_data_135[target_list[i]]['flux'])/4)))

		for j in range(0,len(target_list),1):
			sig_p.append(np.sqrt((real_q[j]**2*q_errors[j]**2+real_u[j]**2
				*u_errors[j]**2)/(real_q[j]**2+real_u[j]**2)))

		for k in range(0,len(target_list),1):
			theta_errors.append(sig_p[k]/(2*real_p[k]*dtr))
			
		return(q_errors,u_errors,sig_p,flux_sig,theta_errors)   
		
	# Store Q, U and P, Theta and P error values in following arrays
	data_array = parameter_errors(norm_flux_err_0,norm_flux_err_45,
		norm_flux_err_90,norm_flux_err_135,real_p,ordin_data_0,extra_data_0,
		ordin_data_45,extra_data_45,ordin_data_90,extra_data_90,ordin_data_135,
		extra_data_135,target_list)
	q_errors,u_errors,sig_p,flux_sig,theta_errors = data_array

	def estimated_polarisation(ordin_data_0,extra_data_0,ordin_data_45,
	extra_data_45,ordin_data_90,extra_data_90,ordin_data_135,extra_data_135,
	ordin_fluxerr_0,ordin_fluxerr_45,ordin_fluxerr_90,ordin_fluxerr_135,
	extra_fluxerr_0,extra_fluxerr_45,extra_fluxerr_90,extra_fluxerr_135,real_p,
	sig_p,target_list):
		# Calculate etap (rough estimate of flux snr) and then use MAS
		# estimator from Plaszczynski et al. 2015 to correct for bias
		snr_f0 = []
		snr_f45 = []
		snr_f90 = []
		snr_f135 = []
		snr_fav = []
		eta = []
		p_corr = []
		
		for i in range(0,len(target_list),1):
			snr_f0.append((ordin_data_0[target_list[i]]['flux']+
				extra_data_0[target_list[i]]['flux'])/np.sqrt((
				ordin_fluxerr_0[i]**2)+(extra_fluxerr_0[i]**2)))
			snr_f45.append((ordin_data_45[target_list[i]]['flux']+
				extra_data_45[target_list[i]]['flux'])/np.sqrt((
				ordin_fluxerr_45[i]**2)+(extra_fluxerr_45[i]**2)))
			snr_f90.append((ordin_data_90[target_list[i]]['flux']+
				extra_data_90[target_list[i]]['flux'])/np.sqrt((
				ordin_fluxerr_90[i]**2)+(extra_fluxerr_90[i]**2)))
			snr_f135.append((ordin_data_135[target_list[i]]['flux']+
				extra_data_135[target_list[i]]['flux'])/np.sqrt((
				ordin_fluxerr_135[i]**2)+(ordin_fluxerr_135[i]**2)))

		for j in range(0,len(target_list),1):
			snr_fav.append((snr_f0[j]+snr_f45[j]+snr_f90[j]+snr_f135[j])/4)

		for k in range(0,len(target_list),1):
			eta.append(real_p[k]*snr_fav[k])

		for l in range(0,len(target_list),1):
			p_corr.append(real_p[l]-(sig_p[l]**2*(1-np.exp(-(real_p[l]**2/
				sig_p[l]**2)))/(2*real_p[l])))
			
		return(eta,p_corr)

	# Eta, estimator name and corrected P values stored in following arrays
	pol_values = estimated_polarisation(ordin_data_0,extra_data_0,
		ordin_data_45,extra_data_45,ordin_data_90,extra_data_90,ordin_data_135,
		extra_data_135,ordin_fluxerr_0,ordin_fluxerr_45,ordin_fluxerr_90,
		ordin_fluxerr_135,extra_fluxerr_0,extra_fluxerr_45,extra_fluxerr_90,
		ordin_fluxerr_135,real_p,sig_p,target_list)
				 
	eta_values,p_corr_values = pol_values

	# Convert polarisations into percentages (*100)
	q_values = [x * 100 for x in q_values]
	real_q = [x * 100 for x in real_q]
	q_errors = [x * 100 for x in q_errors]
	u_values = [x * 100 for x in u_values]
	real_u = [x * 100 for x in real_u]
	u_errors = [x * 100 for x in u_errors]
	p_values = [x * 100 for x in p_values]
	real_p = [x * 100 for x in real_p]
	sig_p = [x * 100 for x in sig_p]
	p_corr_values = [x * 100 for x in p_corr_values]

	# Round all values to 5 significant figures
	q = []
	q_r = []
	q_err = []
	u = []
	u_r = []
	u_err = []
	p = []
	p_r = []
	sigma_p = []
	p_c = []
	theta = []
	theta_err = []
	eta = []

	for i in range(0,len(target_list),1):
		q.append(round(q_values[i],5))
		q_r.append(round(real_q[i],5))
		q_err.append(round(q_errors[i],5))
		u.append(round(u_values[i],5))
		u_r.append(round(real_u[i],5))
		u_err.append(round(u_errors[i],5))
		p.append(round(p_values[i],5))
		p_r.append(round(real_p[i],5))
		sigma_p.append(round(sig_p[i],5))
		p_c.append(round(p_corr_values[i],5))
		theta.append(round(corr_theta_values[i],5))
		theta_err.append(round(theta_errors[i],5))
		eta.append(round(eta_values[i],5))

	# Write results to file    
	orig_stdout = sys.stdout
	result_file=folder_path+'source_results.txt'
	resultf = open(result_file, "w")
	sys.stdout = resultf
	print('                         ### RESULTS ###                         ')
	print('-----------------------------------------------------------------')
	print('')
	print(' Qm(%)  Qr(%)  Q Err(%)  Um(%)  Ur(%)  U Err(%)')
	
	for i in range(0,len(target_list),1):
		print(q[i],q_r[i],q_err[i],u[i],u_r[i],u_err[i])
		
	print('')
	print('Pm(%)   Pr(%)   SNR   Sig_P(%)   Pcorr(%)')
	
	for j in range(0,len(target_list),1):
		print(p[j],p_r[j],round(p_c[j]/sigma_p[j],5),sigma_p[j],p_c[j])
		
	print('')
	print(' Ang    Ang Err')
	
	for k in range(0,len(target_list),1):
		print(theta[k],theta_err[k])
		
	print('')
	print('-----------------------------------------------------------------')
	sys.stdout = orig_stdout
	resultf.close()

	# Close matplotlib windows
	plt.close('all')
	return 0

def main():
	# Run the script
	folder_path,gain,wave_band,par_ang = get_args()
	return sofi_pol(folder_path,gain,wave_band,par_ang)
	
if __name__ == '__main__':
    sys.exit(main())