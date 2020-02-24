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
import pandas as pd
import os
import sys
import argparse


def get_args():
	""" Parse command line arguments """
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('Directory',metavar='DIR',type=str,action='store',
		help='Required directory')
	parser.add_argument('--gain',type=float,default=1.25,dest='gain',
		help='Manually choose the gain - electrons per ADU (default = 1.25)')
		
	args = parser.parse_args()
	directory = args.__dict__['Directory']
	gain = args.gain
	return directory,gain
	

def fors2_pol(directory,gain):
	""" Calculates polarisation measurments from FORS1 observations.
	Requires the four angle ord and exord files to be in the directory.
	"""

	
	def beam_data(angle_file):
		# Extracts data for all targets per angle of selected beam
		total_data = {}
		
		cols = ['x','y','flux','area','msky','st_dev','n_sky']
		beam_data = pd.read_csv(angle_file,header=0,names=cols,
			delim_whitespace=True)
	
		return beam_data

	
	def flux_error(beam_info,target_list,gain):
		# Calculates the flux uncertainty for each source per angle per beam
		flux_error = []
		k = 1
		nd = 1
		eta = 1
	   
		for i in range(0,len(target_list),1):		
			flux_err1 = beam_info.flux[i]/(gain*eta*nd)
			flux_err2 = (beam_info.area[i]*beam_info.st_dev[i]*
				beam_info.st_dev[i])
			flux_err3 = ((k/beam_info.n_sky[i])*
				(beam_info.area[i]*beam_info.st_dev[i])**2)
			
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
			nf1 = (ordin_beam.flux[i]-extra_beam.flux[i])
			nf2 = (ordin_beam.flux[i]+extra_beam.flux[i])
			norm_flux = nf1/nf2
			norm_flux_value.append(norm_flux)
			a = np.sqrt((ordin_fluxerr[i]**2)+(extra_fluxerr[i]**2))
			b = (ordin_beam.flux[i]-extra_beam.flux[i])
			c = (ordin_beam.flux[i]+extra_beam.flux[i])
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
			flux_sig.append(1/(np.sqrt((ordin_data_0.flux[i]+
				extra_data_0.flux[i]+ordin_data_22.flux[i]+
				extra_data_22.flux[i]+ordin_data_45.flux[i]+
				extra_data_45.flux[i]+ordin_data_67.flux[i]+
				extra_data_67.flux[i])/4)))

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
			snr_f0.append((ordin_data_0.flux[i]+extra_data_0.flux[i])/
				np.sqrt((ordin_fluxerr_0[i]**2)+(extra_fluxerr_0[i]**2)))
			snr_f22.append((ordin_data_22.flux[i]+extra_data_22.flux[i])/
				np.sqrt((ordin_fluxerr_22[i]**2)+(extra_fluxerr_22[i]**2)))
			snr_f45.append((ordin_data_45.flux[i]+extra_data_45.flux[i])/
				np.sqrt((ordin_fluxerr_45[i]**2)+(extra_fluxerr_45[i]**2)))
			snr_f67.append((ordin_data_67.flux[i]+extra_data_67.flux[i])/
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
	target_list = []
	for i in range(0,len(ordin_data_0.x),1):	
		name = 'Source '+ str(i+1)
		target_list.append(name)

	# Ensure all angles in both ordinary and extraordinary beams have the
	# same number of sources
	if (len(ordin_data_0.x) or len(ordin_data_22.x) or len(ordin_data_45.x)
		or len(ordin_data_67.x)) != (len(extra_data_0.x) or
		len(extra_data_22.x) or len(extra_data_45.x) or len(extra_data_67.x)):
		
		print('One or more data files have unequal numbers of sources!')
		sys.exit()
		
	# Calculate and store flux errors
	ordin_fluxerr_0 = flux_error(ordin_data_0,target_list,gain)
	ordin_fluxerr_22 = flux_error(ordin_data_22,target_list,gain)
	ordin_fluxerr_45 = flux_error(ordin_data_45,target_list,gain)
	ordin_fluxerr_67 = flux_error(ordin_data_67,target_list,gain)

	extra_fluxerr_0 = flux_error(extra_data_0,target_list,gain)
	extra_fluxerr_22 = flux_error(extra_data_22,target_list,gain)
	extra_fluxerr_45 = flux_error(extra_data_45,target_list,gain)
	extra_fluxerr_67 = flux_error(extra_data_67,target_list,gain)
	
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
	pol_values = estimated_polarisation(ordin_data_0,extra_data_0,
		ordin_data_22,extra_data_22,ordin_data_45,extra_data_45,ordin_data_67,
		extra_data_67,ordin_fluxerr_0,ordin_fluxerr_22,ordin_fluxerr_45,
		ordin_fluxerr_67,extra_fluxerr_0,extra_fluxerr_22,extra_fluxerr_45,
		extra_fluxerr_67,p_values,sig_p,target_list)

	eta_values,p_corr_values = pol_values

	# Convert pol values into percentages and round to 5 s.f
	q_values = [round(x*100,5) for x in q_values]
	q_errors = [round(x*100,5) for x in q_errors]
	u_values = [round(x*100,5) for x in u_values]
	u_errors = [round(x*100,5) for x in u_errors]
	p_values = [round(x*100,5) for x in p_values]
	sig_p = [round(x*100,5) for x in sig_p]
	p_corr_values = [round(x*100,5) for x in p_corr_values]
	corr_theta_values = [round(x,5) for x in corr_theta_values]
	theta_errors = [round(x,5) for x in theta_errors]
	eta_values = [round(x,5) for x in eta_values]
	snr = [round(x/y,5) for x, y in zip(p_corr_values,sig_p)]
	
	# Create dataframes and save results to file
	cols = ['Qm(%)','Q Err(%)','Um(%)','U Err(%)','Pm(%)','SNR','Sig_P(%)',
		'Pcorr(%)','Angle','Angle Err']
	df = pd.DataFrame({cols[0]:q_values,cols[1]:q_errors,cols[2]:u_values,
		cols[3]:u_errors,cols[4]:p_values,cols[5]:snr,cols[6]:sig_p,
		cols[7]:p_corr_values,cols[8]:corr_theta_values,cols[9]:theta_errors})
	df.to_string(folder_path+'source_results.txt',index=False,justify='left')
	return 0
	

def main():
	""" Run Script """
	directory,gain = get_args()
	return fors2_pol(directory,gain)
	
	
if __name__ == '__main__':
    sys.exit(main())