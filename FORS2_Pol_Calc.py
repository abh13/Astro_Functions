#!/usr/bin/python
# -*- coding: utf-8 -*-
# FORS1 polarisation script v1.0
# Created by Adam Higgins
# Email: abh13@le.ac.uk

__doc__ = """ This script calculates the polarisation measurements for FORS2
observations. Please make sure the eight input files follow the naming
convention - angleXXX_ord.txt/angleXXX_exord.txt
"""

import numpy as np
import os
import sys
import argparse


def get_args():
	""" Parse command line arguments """
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("Directory",metavar="DIR",type=str,action="store",
		help="Required directory")
		
	args = parser.parse_args()
	directory = args.__dict__['Directory']
	return directory
	

def fors2_pol(directory):
	""" Calculates polarisation measurments from FORS1 observations.
	Requires the four angle ord and exord files to be in the directory.
	"""

	
	def beam_data(beam_angle):
		# Extracts data for all targets per angle of selected beam	
		total_data = {}
		target_list = []
		
		f = open(beam_angle,'r')
		data = np.genfromtxt(beam_angle,delimiter='',dtype=float,skip_header=1,
			usecols=[1,2,3,4,5,6,7],unpack=True)
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
				'flux': fluxbgs[i], 'area': sourcearea[i],
				'msky': meanbg[i], 'st dev': bgstd[i],'n sky': bgarea[i]}
				
		f.close()
		return total_data

	
	def target_list(data_list):
		# Creates a target list with main target and relevant
		# number of field stars
		target_list = []

		for i in range(0,len(data_list),1):		
			name = 'Source '+ str(i+1)
			target_list.append(name)

		return target_list

	
	def flux_error(beam_info,target_list):
		# Calculates the flux uncertainty for each source per angle per beam
		flux_error = []
		gain = 1.25
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
			a = np.sqrt((ordin_fluxerr[i]**2)+(extra_fluxerr[i]**2))
			b = (ordin_beam[target_list[i]]['flux']-
				extra_beam[target_list[i]]['flux'])
			c = (ordin_beam[target_list[i]]['flux']+
				extra_beam[target_list[i]]['flux'])
			norm_flux_e = norm_flux*np.sqrt(((a/b)**2)+((a/c)**2))
			norm_flux_err.append(norm_flux_e)

		return(norm_flux_value,norm_flux_err)

	
	def pol_param(norm_flux_0,norm_flux_22,norm_flux_45,norm_flux_67,
		target_list):
		# Calculates the measured Q, U, and P values of the objects
		q_values = []
		u_values = []
		p_values = []
		dtr = np.pi/180

		for i in range(0,len(target_list),1):	
			q = ((0.5*norm_flux_0[i]*np.cos(4*0*dtr))+(0.5*norm_flux_22[i]*
				np.cos(4*22.5*dtr))+(0.5*norm_flux_45[i]*np.cos(4*45*dtr))+
				(0.5*norm_flux_67[i]*np.cos(4*67.5*dtr)))
			u = ((0.5*norm_flux_0[i]*np.sin(4*0*dtr))+(0.5*norm_flux_22[i]*
				np.sin(4*22.5*dtr))+(0.5*norm_flux_45[i]*np.sin(4*45*dtr))+
				(0.5*norm_flux_67[i]*np.sin(4*67.5*dtr)))
			p = np.sqrt((q**2)+(u**2))
			q_values.append(q)
			u_values.append(u)
			p_values.append(p)
			
		return(q_values,u_values,p_values)
			
	
	def calc_theta(u,q):
		# Calculates theta for all objects
		theta_values = []
		dtr = np.pi/180

		for i in range(0,len(target_list),1):		
			theta = 0.5*np.arctan(u[i]/q[i])*(1/dtr)
			theta_values.append(theta)
			
		return theta_values
		

	def position_angle(theta_values,q,u,target_list):
		# Calculate proper position angles
		corr_theta_values = []
		
		for i in range(0,len(target_list),1):

			if q[i] < 0:
				corr_theta_values.append(theta_values[i]+90)

			if q[i] > 0 and u[i] > 0:
				corr_theta_values.append(theta_values[i]+0)

			if q[i] > 0 and u[i] < 0:
				corr_theta_values.append(theta_values[i]+180)

		return corr_theta_values
		

	def parameter_errors(norm_flux_err_0,norm_flux_err_22,norm_flux_err_45,
		norm_flux_err_67,p_values,ordin_data_0,extra_data_0,ordin_data_22,
		extra_data_22,ordin_data_45,extra_data_45,ordin_data_67,extra_data_67,
		target_list):
		# Calculate errors on Q, U, P, Theta, SD of the average flux per angle
		q_errors = []
		u_errors = []
		sig_p = []
		flux_sig = []
		theta_errors = []
		dtr = np.pi/180

		for i in range(0,len(target_list),1):		
			q_errors.append(np.sqrt(((0.5*norm_flux_err_0[i]*np.cos(4*0*dtr))
				**2)+((0.5*norm_flux_err_22[i]*np.cos(4*22.5*dtr))**2)+
				((0.5*norm_flux_err_45[i]*np.cos(4*45*dtr))**2)+
				((0.5*norm_flux_err_67[i]*np.cos(4*67.5*dtr))**2)))
			u_errors.append(np.sqrt(((0.5*norm_flux_err_0[i]*np.sin(4*0*dtr))
				**2)+((0.5*norm_flux_err_22[i]*np.sin(4*22.5*dtr))**2)+
				((0.5*norm_flux_err_45[i]*np.sin(4*45*dtr))**2)+
				((0.5*norm_flux_err_67[i]*np.sin(4*67.5*dtr))**2)))
			flux_sig.append(1/(np.sqrt((ordin_data_0[target_list[i]]['flux']+
				extra_data_0[target_list[i]]['flux']+
				ordin_data_22[target_list[i]]['flux']+
				extra_data_22[target_list[i]]['flux']+
				ordin_data_45[target_list[i]]['flux']+
				extra_data_45[target_list[i]]['flux']+
				ordin_data_67[target_list[i]]['flux']+
				extra_data_67[target_list[i]]['flux'])/4)))

		for j in range(0,len(target_list),1):
			sig_p.append(np.sqrt((q_values[j]**2*q_errors[j]**2+u_values[j]**2
				*u_errors[j]**2)/(q_values[j]**2+u_values[j]**2)))

		for k in range(0,len(target_list),1):
			theta_errors.append(sig_p[k]/(2*p_values[k]*dtr))

		return(q_errors,u_errors,sig_p,flux_sig,theta_errors)  
		

	def estimated_polarisation(ordin_data_0,extra_data_0,ordin_data_22,
		extra_data_22,ordin_data_45,extra_data_45,ordin_data_67,extra_data_67,
		ordin_fluxerr_0,ordin_fluxerr_22,ordin_fluxerr_45,ordin_fluxerr_67,
		extra_fluxerr_0,extra_fluxerr_22,extra_fluxerr_45,extra_fluxerr_67,
		p_values,sig_p,target_list):
		# Calculate etap (rough estimate of flux snr) and then use MAS
		# estimator from Plaszczynski et al. 2015 to correct for bias		
		snr_f0 = []
		snr_f22 = []
		snr_f45 = []
		snr_f67 = []
		snr_fav = []
		eta = []
		p_corr = []
		
		for i in range(0,len(target_list),1):
			snr_f0.append((ordin_data_0[target_list[i]]['flux']+
				extra_data_0[target_list[i]]['flux'])/
				np.sqrt((ordin_fluxerr_0[i]**2)+(extra_fluxerr_0[i]**2)))
			snr_f22.append((ordin_data_22[target_list[i]]['flux']+
				extra_data_22[target_list[i]]['flux'])/
				np.sqrt((ordin_fluxerr_22[i]**2)+(extra_fluxerr_22[i]**2)))
			snr_f45.append((ordin_data_45[target_list[i]]['flux']+
				extra_data_45[target_list[i]]['flux'])/
				np.sqrt((ordin_fluxerr_45[i]**2)+(extra_fluxerr_45[i]**2)))
			snr_f67.append((ordin_data_67[target_list[i]]['flux']+
				extra_data_67[target_list[i]]['flux'])/
				np.sqrt((ordin_fluxerr_67[i]**2)+(extra_fluxerr_67[i]**2)))

		for j in range(0,len(target_list),1):
			snr_fav.append((snr_f0[j]+snr_f22[j]+snr_f45[j]+snr_f67[j])/4)

		for k in range(0,len(target_list),1):
			eta.append(p_values[k]*snr_fav[k])

		for l in range(0,len(target_list),1):
			p_corr.append(p_values[l]-(sig_p[l]**2*(1-np.exp(-(p_values[l]**2/
				sig_p[l]**2)))/(2*p_values[l])))
						   
		return(eta,p_corr)

		
	# Begin by reading in files
	folder_path = directory
	file_name_ord0 = 'angle0_ord.txt'
	file_name_ord22 = 'angle225_ord.txt'
	file_name_ord45 = 'angle45_ord.txt'
	file_name_ord67 = 'angle675_ord.txt'
	file_name_ext0 = 'angle0_exord.txt'
	file_name_ext22 = 'angle225_exord.txt'
	file_name_ext45 = 'angle45_exord.txt'
	file_name_ext67 = 'angle675_exord.txt'

	ordin_0 = os.path.join(folder_path,file_name_ord0)
	ordin_22 = os.path.join(folder_path,file_name_ord22)
	ordin_45 = os.path.join(folder_path,file_name_ord45)
	ordin_67 = os.path.join(folder_path,file_name_ord67)
	extra_0 = os.path.join(folder_path,file_name_ext0)
	extra_22 = os.path.join(folder_path,file_name_ext22)
	extra_45 = os.path.join(folder_path,file_name_ext45)
	extra_67 = os.path.join(folder_path,file_name_ext67)

	# Defines two lists of ordinary and extra ordinary files to extract data
	ordinary_beam = [ordin_0,ordin_22,ordin_45,ordin_67]
	extra_beam = [extra_0,extra_22,extra_45,extra_67]
	
	# Raise Error if files or folder cannot be found
	try:
		ordin_data_0 = beam_data(ordinary_beam[0])
		ordin_data_22 = beam_data(ordinary_beam[1])
		ordin_data_45 = beam_data(ordinary_beam[2])
		ordin_data_67 = beam_data(ordinary_beam[3])
		extra_data_0 = beam_data(extra_beam[0])
		extra_data_22 = beam_data(extra_beam[1])
		extra_data_45 = beam_data(extra_beam[2])
		extra_data_67 = beam_data(extra_beam[3])
		
	except FileNotFoundError as e:
		print('Cannot find the folder or files you are looking for')
		sys.exit()

	# Creates target list of sources
	target_list = target_list(ordin_data_0)

	# Ensure all angles in both ordinary and extraordinary beams have the
	# same number of sources
	if (len(ordin_data_0) or len(ordin_data_22) or len(ordin_data_45) or 
		len(ordin_data_67)) != (len(extra_data_0) != len(extra_data_22) or 
		len(extra_data_45) or len(extra_data_67)):
		
		print('One or more data files have unequal numbers of sources!')
		sys.exit()
		
	# Calculate and store flux errors
	ordin_fluxerr_0 = flux_error(ordin_data_0,target_list)
	ordin_fluxerr_22 = flux_error(ordin_data_22,target_list)
	ordin_fluxerr_45 = flux_error(ordin_data_45,target_list)
	ordin_fluxerr_67 = flux_error(ordin_data_67,target_list)

	extra_fluxerr_0 = flux_error(extra_data_0,target_list)
	extra_fluxerr_22 = flux_error(extra_data_22,target_list)
	extra_fluxerr_45 = flux_error(extra_data_45,target_list)
	extra_fluxerr_67 = flux_error(extra_data_67,target_list)
	
	# Calculate and store normalised flux values and errors
	norm_flux_0,norm_flux_err_0 = norm_flux(ordin_data_0,extra_data_0,
		ordin_fluxerr_0,extra_fluxerr_0,target_list)
	norm_flux_22,norm_flux_err_22 = norm_flux(ordin_data_22,extra_data_22,
		ordin_fluxerr_22,extra_fluxerr_22,target_list)
	norm_flux_45,norm_flux_err_45 = norm_flux(ordin_data_45,extra_data_45,
		ordin_fluxerr_45,extra_fluxerr_45,target_list)
	norm_flux_67,norm_flux_err_67 = norm_flux(ordin_data_67,extra_data_67,
		ordin_fluxerr_67,extra_fluxerr_67,target_list)
		
	# Calculate and store Q, U, and P values
	q_values,u_values,p_values = pol_param(norm_flux_0,norm_flux_22,
		norm_flux_45,norm_flux_67,target_list)
		
	# Calculate and store theta values
	theta_values = calc_theta(u_values,q_values)
	corr_theta_values = position_angle(theta_values,q_values,u_values,
		target_list)
	
	# Calculate errors on Q, U and P, theta and flux P
	data_array = parameter_errors(norm_flux_err_0,norm_flux_err_22,
		norm_flux_err_45,norm_flux_err_67,p_values,ordin_data_0,
		extra_data_0,ordin_data_22,extra_data_22,ordin_data_45,
		extra_data_45,ordin_data_67,extra_data_67,target_list)
				 
	q_errors,u_errors,sig_p,flux_sig,theta_errors = data_array

	# Calculate eta and corrected P values
	pol_values = estimated_polarisation(ordin_data_0,extra_data_0,ordin_data_22,
		extra_data_22,ordin_data_45,extra_data_45,ordin_data_67,
		extra_data_67,ordin_fluxerr_0,ordin_fluxerr_22,ordin_fluxerr_45,
		ordin_fluxerr_67,extra_fluxerr_0,extra_fluxerr_22,extra_fluxerr_45,
		extra_fluxerr_67,p_values,sig_p,target_list)

	eta_values,p_corr_values = pol_values

	# Convert pol values into percentages
	q_values = [x * 100 for x in q_values]
	q_errors = [x * 100 for x in q_errors]
	u_values = [x * 100 for x in u_values]
	u_errors = [x * 100 for x in u_errors]
	p_values = [x * 100 for x in p_values]
	sig_p = [x * 100 for x in sig_p]
	p_corr_values = [x * 100 for x in p_corr_values]     
		
	# Round all values to 5 significant figures
	q = []
	q_err = []
	u = []
	u_err = []
	p = []
	sigma_p = []
	p_corr = []
	theta = []
	theta_err = []
	eta = []

	for i in range(0,len(target_list),1):
		q.append(round(q_values[i],5))
		q_err.append(round(q_errors[i],5))
		u.append(round(u_values[i],5))
		u_err.append(round(u_errors[i],5))
		p.append(round(p_values[i],5))
		sigma_p.append(round(sig_p[i],5))
		p_corr.append(round(p_corr_values[i],5))
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
	print('Qm(%)  Q Err(%)  Um(%)  U Err(%)')
	
	for i in range(0,len(target_list),1):
		print(q[i],q_err[i],u[i],u_err[i])
		
	print('')
	print('Pm(%)    SNR    Sig_P(%)    Pcorr(%)')
	
	for j in range(0,len(target_list),1):
		print(p[j],round(p_corr[j]/sig_p[j],5),sigma_p[j],p_corr[j])
		
	print('')
	print('Ang    Ang Err')
	
	for k in range(0,len(target_list),1):
		print(theta[k],theta_err[k])
		
	print('')
	print('-----------------------------------------------------------------')

	sys.stdout = orig_stdout
	resultf.close()
	return 0
	

def main():
	""" Run Script """
	directory = get_args()
	return fors2_pol(directory)
	
	
if __name__ == '__main__':
    sys.exit(main())