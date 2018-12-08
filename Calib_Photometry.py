#!/usr/bin/python
# -*- coding: utf-8 -*-
# Scrapes APASS, SDSS, PanSTARRs and Skymapper
# Created by Adam Higgins
# Email: abh13@le.ac.uk

__doc__ = """ Calculate the corrected magnitudes (AB) for sources within an
image fits file. This is accomplished by comparing field star observed
magnitudes against measured catalogue values and determines the offset. The
script currently uses the APASS, SDSS PanSTARRs and Skymapper catalogues. It
can calibrate images in the Johnson-Cousins filters (B,V,R) and SDSS
filters (u,g,r,i,z).
"""

__version__ = '1.5'

from scipy.optimize import curve_fit
from bs4 import BeautifulSoup
from astropy.wcs import WCS
from astropy.io import fits
from matplotlib.patches import Circle
import sep
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import requests


def get_args():
	### Parse Arguments
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('Image File',metavar='FILE',type=str,action='store',
		help='Name of the image fits file (xxxx.fits)')
	parser.add_argument('--g',type=float,default=1.1,dest='gain',
		help='Gain (default = 1.1)')
	parser.add_argument('--d',type=str,default=os.getcwd(),dest='directory',
		help='Desired directory (default = current directory)')
	parser.add_argument('--sr',type=float,default=5,dest='searchrad',
		help='Search radius in arcmin (default = 5)')
	parser.add_argument('--f',type=str,default='V',dest='filter',
		help='The filter used for the observations. Choices are\
		either Johnson B,V,R or SDSS u,g,r,i,z (default = V)')
	parser.add_argument('--sig',type=float,default=3,dest='sigma',
		help='Number of sigma to use for clipping outlier stars\
		during the magnitude offset calibration (default = 3)')
	parser.add_argument('--ap',type=float,default=3,dest='aperture',
		help='Radius of aperture for photometry in pixels (default = 5)')
		
	args = parser.parse_args()
	directory = args.directory
	gain = args.gain
	searchrad = args.searchrad
	waveband = args.filter
	im_file = args.__dict__['Image File']
	sigma = args.sigma
	aperture = args.aperture
	
	# Check for valid filters
	if (waveband != 'V' and waveband != 'B' and waveband != 'R' and
		waveband != 'u' and waveband != 'i' and waveband != 'g'
		and waveband != 'r'	and waveband != 'z'):
		
		print('Script does not calibrate for this waveband!')
		sys.exit()
		
	return directory,gain,searchrad,waveband,im_file,sigma,aperture
	

def im_phot(directory,gain,im_file,aperture):
	# Perform photometry on the image

	### Read in fits image file and create array with wcs coordinates
	os.chdir(directory)
	hdulist = fits.open(im_file)
	w = WCS(im_file)
	data = hdulist[0].data
	data[np.isnan(data)] = 0
	hdulist.close()

	### Calculate center point of image (RA, Dec) if not input by user
	targetra, targetdec = w.all_pix2world(len(data[:,0])/2,len(data[0,:])/2,0)

	### Use SEP for background subtraction and source detection
	datasw = data.byteswap().newbyteorder().astype('float64')
	bkg = sep.Background(datasw)
	data_bgs = data - bkg
	data_bgs[data_bgs < 0] = 0
	mean = np.mean(data_bgs)
	median = np.median(data_bgs)
	std = bkg.globalrms
	objects = sep.extract(data_bgs,3,err=bkg.globalrms)
	objra, objdec = w.all_pix2world(objects['x'],objects['y'],0)

	### Find dummy magnitudes using aperture photometry and plot images
	fig, ax = plt.subplots()
	image = plt.imshow(data_bgs,cmap='gray',vmin=(mean-3*std),
		vmax=(mean+3*std),origin='lower')
	sepmag = []
	sepmagerr = []
	ra = []
	dec = []
	xpixel = []
	ypixel = []

	for i in range(len(objects)):
				
		# Perform circular aperture photometry
		flux,fluxerr,flag = sep.sum_circle(data_bgs,objects['x'][i],
			objects['y'][i],aperture,err=std,gain=gain)
		mag = -2.5*np.log10(flux)
		maglimit1 = -2.5*np.log10((flux+fluxerr))
		maglimit2 = -2.5*np.log10((flux-fluxerr))
		magerr1 = np.abs(mag-maglimit1)
		magerr2 = np.abs(mag-maglimit2)
		magerr = (magerr1+magerr2)/2
		
		## Save object properties to arrays
		sepmag.append(mag)
		sepmagerr.append(magerr)
		ra.append(objra[i])
		dec.append(objdec[i])
		xpixel.append(objects['x'][i])
		ypixel.append(objects['y'][i])
		
		## Plot the detections on the image
		out = Circle(xy=(objects['x'][i],objects['y'][i]),radius=aperture)
		out.set_facecolor('none')
		out.set_edgecolor('red')
		ax.add_artist(out)

	plt.savefig(directory+'detections.png')	
	return targetra,targetdec,sepmag,sepmagerr,ra,dec,xpixel,ypixel


def apass_search(searchrad,waveband,targetra,targetdec):
	# Search for all stars within search radius of target in APASS
	# catalogue
	
	# Set up url and arrays
	sr_deg = float(searchrad*0.0166667)
	star_ra = []
	star_dec = []
	star_mag = []
	star_magerr = []
	apass_url1 = 'https://www.aavso.org/cgi-bin/apass_download.pl?'
	apass_url2 = 'ra={0}&dec={1}&radius={2}&outtype=0'
	apass_url = apass_url1 + apass_url2
	apass_url = apass_url.format(targetra,targetdec,sr_deg)
	
	# Attempt to parse url to find stars within search radius of filter
	try:
		apass_r = requests.get(apass_url,timeout=20).text
		apass_soup = BeautifulSoup(apass_r,'lxml')
		apass_table = apass_soup.find('table')
		apass_rows = apass_table.findAll('tr')
		
		for row in apass_rows:		
			try:
				cols = row.findAll('td')
				cols = [ele.text.strip() for ele in cols]
				
				if waveband == 'V':
					star_mag.append(float(cols[5]))
					star_magerr.append(float(cols[6]))
					star_ra.append(float(cols[0]))
					star_dec.append(float(cols[2]))
					
				if waveband == 'B':
					star_mag.append(float(cols[7])-0.09)
					star_magerr.append(float(cols[8]))
					star_ra.append(float(cols[0]))
					star_dec.append(float(cols[2]))
					
				if waveband == 'R':
					g_mag = float(cols[9])
					g_magerr = float(cols[10])
					r_mag = float(cols[11])
					r_magerr = float(cols[12])
					R_mag = r_mag - 0.1837 * (g_mag - r_mag) - 0.0971
					R_magerr = (np.sqrt((1.1837*r_magerr)**2
						+(0.1837*g_magerr)**2))
					star_mag.append(round(R_mag,6))
					star_magerr.append(round(R_magerr,6))
					star_ra.append(float(cols[0]))
					star_dec.append(float(cols[2]))
					
				if waveband == 'g':
					star_mag.append(float(cols[9]))
					star_magerr.append(float(cols[10]))
					star_ra.append(float(cols[0]))
					star_dec.append(float(cols[2]))
					
				if waveband == 'r':
					star_mag.append(float(cols[11]))
					star_magerr.append(float(cols[12]))
					star_ra.append(float(cols[0]))
					star_dec.append(float(cols[2]))
					
				if waveband == 'i':
					star_mag.append(float(cols[13]))
					star_magerr.append(float(cols[14]))
					star_ra.append(float(cols[0]))
					star_dec.append(float(cols[2]))
					
			except ValueError:
				continue
	
	# Raise error if something goes wrong
	except requests.exceptions.RequestException as e:
		print('\nException raised for APASS url')
		print(e)
		print('')
	
	# Create list with catalogue name
	star_cat = ['APASS'] * len(star_ra)	
	return star_ra,star_dec,star_mag,star_magerr,star_cat


def sdss_search(searchrad,waveband,targetra,targetdec):	
	# Search for all stars within search radius of target in SDSS
	# catalogue
	
	# set up url, arrays and number of returned results
	star_ra = []
	star_dec = []
	star_mag = []
	star_magerr = []
	numberofresults = 999
	sdss_u1 = 'http://skyserver.sdss.org/dr13/en/tools/search/x_results.aspx?'
	sdss_u2 = 'searchtool=Radial&uband=&gband=&rband=&iband=&zband=&jband=&'
	sdss_u3 = 'hband=&kband=&TaskName=Skyserver.Search.Radial&ReturnHtml=true'
	sdss_u4 = '&whichphotometry=optical&coordtype=equatorial&ra={0}&dec={1}&'
	sdss_u5 = 'radius={2}&min_u=0&max_u=20&min_g=0&max_g=20&min_r=0&max_r=20&'
	sdss_u6 = 'min_i=0&max_i=20&min_z=0&max_z=20&min_j=0&max_j=20&min_h=0&'
	sdss_u7 = 'max_h=20&min_k=0&max_k=20&format=csv&TableName=&limit={3}'
	sdss_url = sdss_u1+sdss_u2+sdss_u3+sdss_u4+sdss_u5+sdss_u6+sdss_u7
	sdss_url = sdss_url.format(targetra,targetdec,searchrad,numberofresults)
	
	# Attempt to parse url to find stars within search radius of filter
	try:
		sdss_text = requests.get(sdss_url,timeout=20).text
		lines = sdss_text.strip().split('\n')
		c = 0
		
		for line in lines:	
			if c > 2:
				cols = line.split(',')
				obj_type = float(cols[6])
				
				if obj_type == 6:
					star_ra.append(round(float(cols[7]),8))
					star_dec.append(round(float(cols[8]),7))
					g_mag = float(cols[10])
					r_mag = float(cols[11])
					g_magerr = float(cols[15])
					r_magerr = float(cols[16])
					
					if waveband == 'V':
						V_mag = g_mag - 0.5784 * (g_mag - r_mag) - 0.0038
						star_mag.append(round(V_mag,6))
						V_magerr = (np.sqrt((0.5784*r_magerr)**2+
							(0.4216*g_magerr)**2))
						star_magerr.append(round(V_magerr,6))
						
					if waveband == 'B':
						B_mag = g_mag + 0.3130 * (g_mag - r_mag) + 0.2271
						star_mag.append(round(B_mag,6))
						B_magerr = (np.sqrt((0.3130*r_magerr)**2+
							(1.3130*g_magerr)**2))
						star_magerr.append(round(B_magerr,6))
						
					if waveband == 'R':
						R_mag = r_mag - 0.1837 * (g_mag - r_mag) - 0.0971
						star_mag.append(round(R_mag,6))
						R_magerr = (np.sqrt((1.1837*r_magerr)**2+
							(0.1837*g_magerr)**2))
						star_magerr.append(round(R_magerr,6))
						
					if waveband == 'u':
						star_mag.append(round(float(cols[9]),6)-0.04)
						star_magerr.append(round(float(cols[14]),6))
						
					if waveband == 'g':
						star_mag.append(round(g_mag,6))
						star_magerr.append(round(g_magerr,6))
						
					if waveband == 'r':
						star_mag.append(round(r_mag,6))
						star_magerr.append(round(r_magerr,6))
						
					if waveband == 'i':
						star_mag.append(round(float(cols[12]),6))
						star_magerr.append(round(float(cols[17]),6))
						
					if waveband == 'z':
						star_mag.append(round(float(cols[13]),6)+0.02)
						star_magerr.append(round(float(cols[18]),6))
			c += 1
	
	# Raise error if something goes wrong
	except requests.exceptions.RequestException as e:
		print('\nException raised for SDSS url!')
		print(e)
		print('')

	# Create list with catalogue name
	star_cat = ['SDSS'] * len(star_ra)	
	return star_ra,star_dec,star_mag,star_magerr,star_cat


def panstarrs_search(searchrad,waveband,targetra,targetdec):	
	# Search for all stars within search radius of target in PanSTARRs
	# catalogue
	
	# Set up arrays and url
	star_ra = []
	star_dec = []
	star_mag = []
	star_magerr = []
	psra = []
	psdec = []
	gmag_psf = []
	gmagerr_psf = []
	gmag_kron = []
	gmagerr_kron = []
	gmag_aper = []
	gmagerr_aper = []
	rmag_psf = []
	rmagerr_psf = []
	rmag_kron = []
	rmagerr_kron = []
	rmag_aper = []
	rmagerr_aper = []
	imag_psf = []
	imagerr_psf = []
	imag_kron = []
	imagerr_kron = []
	imag_aper = []
	imagerr_aper = []
	zmag_psf = []
	zmagerr_psf = []
	zmag_kron = []
	zmagerr_kron = []
	zmag_aper = []
	zmagerr_aper = []
	numberofresults = 999
	sr_deg = float(searchrad*0.0166667)
	ps_url1 = 'http://archive.stsci.edu/panstarrs/search.php?&max_records={0}'
	ps_url2 = '&RA={1}&DEC={2}&SR={3}'
	ps_url = ps_url1 + ps_url2
	ps_url = ps_url.format(numberofresults,targetra,targetdec,sr_deg)
	
	# Attempt to parse url to find stars within search radius of filter
	try:
		pc = 0
		pstable = requests.get(ps_url,timeout=30).text
		
		for lines in pstable.split('<TR>'):		
			pc += 1
			
			if pc >= 2:
				columns = re.split("<TD>|</TD>|\n",lines)
				psra.append(float(columns[6]))
				psdec.append(float(columns[8]))
				gmag_psf.append(float(columns[50]))
				gmagerr_psf.append(float(columns[52]))
				gmag_kron.append(float(columns[54]))
				gmagerr_kron.append(float(columns[56]))
				gmag_aper.append(float(columns[58]))
				gmagerr_aper.append(float(columns[60]))
				rmag_psf.append(float(columns[66]))
				rmagerr_psf.append(float(columns[68]))
				rmag_kron.append(float(columns[70]))
				rmagerr_kron.append(float(columns[72]))
				rmag_aper.append(float(columns[74]))
				rmagerr_aper.append(float(columns[76]))
				imag_psf.append(float(columns[82]))
				imagerr_psf.append(float(columns[84]))
				imag_kron.append(float(columns[86]))
				imagerr_kron.append(float(columns[88]))
				imag_aper.append(float(columns[90]))
				imagerr_aper.append(float(columns[92]))
				zmag_psf.append(float(columns[98]))
				zmagerr_psf.append(float(columns[100]))
				zmag_kron.append(float(columns[102]))
				zmagerr_kron.append(float(columns[104]))
				zmag_aper.append(float(columns[106]))
				zmagerr_aper.append(float(columns[108]))
	
	# Raise error if something goes wrong
	except requests.exceptions.RequestException as e:
		print ('\nException raised for PanSTARRs url!!')
		print (e)
		print ('')
	
	# Save parsed star properties for a given filter and remove extended
	# shaped sources
	for i in range(len(psra)):
	
		if (gmag_psf[i] != -999.000 and gmag_kron[i] != -999.000 and
			gmag_aper[i] != -999.000 and rmag_psf[i] != -999.000 and 
			rmag_kron[i] != -999.000 and rmag_aper[i] != -999.000):		
			
			if (np.abs(gmag_psf[i] - gmag_kron[i]) < 0.25 and 
				np.abs(gmag_psf[i] - gmag_aper[i]) < 0.25 and 
				np.abs(gmag_aper[i] - gmag_kron[i]) < 0.25 and
				np.abs(rmag_psf[i] - rmag_kron[i]) < 0.25 and 
				np.abs(rmag_psf[i] - rmag_aper[i]) < 0.25 and 
				np.abs(rmag_aper[i] - rmag_kron[i]) < 0.25):
			
				if waveband == 'V':
					V_mag = (gmag_aper[i]-0.5784*(gmag_aper[i]-rmag_aper[i])
						-0.0038)
					V_magerr = (np.sqrt((0.5784*rmagerr_aper[i])**2+
						(0.4216*gmagerr_aper[i])**2))
					star_mag.append(V_mag)
					star_magerr.append(V_magerr)
					star_ra.append(psra[i])
					star_dec.append(psdec[i])
					
				if waveband == 'B':
					B_mag = (gmag_aper[i]+0.3130*(gmag_aper[i]-rmag_aper[i])
						+0.2271)
					B_magerr = (np.sqrt((0.3130*rmagerr_aper[i])**2+
						(1.3130*gmagerr_aper[i])**2))
					star_mag.append(B_mag)
					star_magerr.append(B_magerr)
					star_ra.append(psra[i])
					star_dec.append(psdec[i])
					
				if waveband == 'R':
					R_mag = (rmag_aper[i]-0.1837*(gmag_aper[i]-rmag_aper[i])
						-0.0971)
					R_magerr = (np.sqrt((1.1837*rmagerr_aper[i])**2+
						(0.1837*gmagerr_aper[i])**2))
					star_mag.append(R_mag)
					star_magerr.append(R_magerr)
					star_ra.append(psra[i])
					star_dec.append(psdec[i])
					
		if waveband == 'g':
			
			if (gmag_psf[i] != -999.000 and gmag_kron[i] != -999.000 and 
				gmag_aper[i] != -999.000):
				
				if (np.abs(gmag_psf[i] - gmag_kron[i]) < 0.25 and 
					np.abs(gmag_psf[i] - gmag_aper[i]) < 0.25 and 
					np.abs(gmag_aper[i] - gmag_kron[i]) < 0.25):
					
					star_mag.append(gmag_aper[i])
					star_magerr.append(gmagerr_aper[i])
					star_ra.append(psra[i])
					star_dec.append(psdec[i])

		if waveband == 'r':
			
			if (rmag_psf[i] != -999.000 and rmag_kron[i] != -999.000 and
				rmag_aper[i] != -999.000):
				
				if (np.abs(rmag_psf[i] - rmag_kron[i]) < 0.25 and
					np.abs(rmag_psf[i] - rmag_aper[i]) < 0.25 and
					np.abs(rmag_aper[i] - rmag_kron[i]) < 0.25):
					
					star_mag.append(rmag_aper[i])
					star_magerr.append(rmagerr_aper[i])
					star_ra.append(psra[i])
					star_dec.append(psdec[i])
				
		if waveband == 'i' :           
			
			if (imag_psf[i] != -999.000 and imag_kron[i] != -999.000 and
				imag_aper[i] != -999.000):
				
				if (np.abs(imag_psf[i] - imag_kron[i]) < 0.25 and 
					np.abs(imag_psf[i] - imag_aper[i]) < 0.25 and 
					np.abs(imag_aper[i] - imag_kron[i]) < 0.25):
					
					star_ra.append(psra[i])
					star_dec.append(psdec[i])
					star_mag.append(imag_aper[i])
					star_magerr.append(imagerr_aper[i])

		if waveband == 'z' :           
			
			if (zmag_psf[i] != -999.000 and zmag_kron[i] != -999.000 and
				zmag_aper[i] != -999.000):
				
				if (np.abs(zmag_psf[i] - zmag_kron[i]) < 0.25 and
					np.abs(zmag_psf[i] - zmag_aper[i]) < 0.25 and
					np.abs(zmag_aper[i] - zmag_kron[i]) < 0.25):
					
					star_ra.append(psra[i])
					star_dec.append(psdec[i])
					star_mag.append(zmag_aper[i])
					star_magerr.append(zmagerr_aper[i])
					
	# Create list with catalogue name
	star_cat = ['PanSTARRs'] * len(star_ra)	
	return star_ra,star_dec,star_mag,star_magerr,star_cat


def skymapper_search(searchrad,waveband,targetra,targetdec):	
	# Search for all stars within search radius of target in Skymapper
	# catalogue	
	
	# set up arrays and url
	star_ra = []
	star_dec = []
	star_mag = []
	star_magerr = []
	sky_ra = []
	sky_dec = []
	sky_u_petro = []
	sky_u_petro_err = []
	sky_u_psf = []
	sky_u_psf_err = []
	sky_v_petro = []
	sky_v_petro_err = []
	sky_v_psf = []
	sky_v_psf_err = []
	sky_g_petro = []
	sky_g_petro_err = []
	sky_g_psf = []
	sky_g_psf_err = []
	sky_r_petro = []
	sky_r_petro_err = []
	sky_r_psf = []
	sky_r_psf_err = []
	sky_i_petro = []
	sky_i_petro_err = []
	sky_i_psf = []
	sky_i_psf_err = []
	sky_z_petro = []
	sky_z_petro_err = []
	sky_z_psf = []
	sky_z_psf_err = []
	sr_deg = float(searchrad*0.0166667)
	sky_url	= "http://skymapper.anu.edu.au/sm-cone/query?RA={0}&DEC={1}&SR={2}"
	sky_url = sky_url.format(targetra,targetdec,sr_deg)
	
	# Attempt to parse url to find stars within search radius of filter
	try:
		skytable = requests.get(sky_url,timeout=30).text
		sc = 0
		
		for lines in skytable.split('<TR>'):		
			sc += 1
			
			if sc >= 2:
				columns = re.split("<TD>|</TD>|\n",lines)
				sky_ra.append(columns[5])
				sky_dec.append(columns[7])
				sky_u_petro.append(columns[33])
				sky_u_petro_err.append(columns[35])
				sky_u_psf.append(columns[29])
				sky_u_psf_err.append(columns[31])
				sky_v_petro.append(columns[41])
				sky_v_petro_err.append(columns[43])
				sky_v_psf.append(columns[37])
				sky_v_psf_err.append(columns[39])
				sky_g_petro.append(columns[49])
				sky_g_petro_err.append(columns[51])
				sky_g_psf.append(columns[45])
				sky_g_psf_err.append(columns[47])
				sky_r_petro.append(columns[57])
				sky_r_petro_err.append(columns[59])
				sky_r_psf.append(columns[53])
				sky_r_psf_err.append(columns[55])
				sky_i_petro.append(columns[65])
				sky_i_petro_err.append(columns[67])
				sky_i_psf.append(columns[61])
				sky_i_psf_err.append(columns[63])
				sky_z_petro.append(columns[73])
				sky_z_petro_err.append(columns[75])
				sky_z_psf.append(columns[69])
				sky_z_psf_err.append(columns[71])
	
	# Raise error if something goes wrong
	except requests.exceptions.RequestException as e:
		print ('\nException raised for Skymapper url!!')
		print (e)
		print ('')
	
	# Save parsed star properties for a given filter and remove extended
	# shaped sources
	for i in range(len(sky_ra)):
	
		if (sky_g_psf[i] != '' and sky_g_petro[i] != '' and
			sky_r_psf[i] != '' and sky_r_petro[i] != ''):
			
			if (np.abs(float(sky_g_psf[i]) - float(sky_g_petro[i])) < 0.25
				and np.abs(float(sky_r_psf[i]) - float(sky_r_petro[i]))
				< 0.25):
			
				if waveband == 'V':
					V_mag = float(sky_g_psf[i])-0.0038
					V_mag = (V_mag-0.5784*(float(sky_g_psf[i])
						-float(sky_r_psf[i])))
					gerr = float(sky_g_psf_err[i])**2
					rerr = float(sky_r_psf_err[i])**2
					V_magerr = np.sqrt((0.5784*rerr)**2+(0.4216*gerr)**2)
					star_mag.append(V_mag)
					star_magerr.append(V_magerr)
					star_ra.append(float(sky_ra[i]))
					star_dec.append(float(sky_dec[i]))
					
				if waveband == 'B':
					B_mag = float(sky_g_psf[i])+0.2271
					B_mag = (B_mag+0.3130*(float(sky_g_psf[i])-
						float(sky_r_psf[i])))
					gerr = float(sky_g_psf_err[i])**2
					rerr = float(sky_r_psf_err[i])**2
					B_magerr = np.sqrt((0.3130*rerr)**2+(1.3130*gerr)**2)
					star_mag.append(B_mag)
					star_magerr.append(B_magerr)
					star_ra.append(float(sky_ra[i]))
					star_dec.append(float(sky_dec[i]))
					
				if waveband == 'R':
					R_mag = float(sky_r_psf[i])-0.0971
					R_mag = (R_mag-0.1837*(float(sky_g_psf[i])-
						float(sky_r_psf[i])))
					gerr = float(sky_g_psf_err[i])**2
					rerr = float(sky_r_psf_err[i])**2
					R_magerr = np.sqrt((1.1837*rerr)**2+(0.1837*gerr)**2)
					star_mag.append(R_mag)
					star_magerr.append(R_magerr)
					star_ra.append(float(sky_ra[i]))
					star_dec.append(float(sky_dec[i]))
					
		if waveband == 'u':
			if (sky_u_psf[i] != '' and sky_u_petro[i] != ''):
				if (np.abs(float(sky_u_psf[i]) - float(sky_u_petro[i]))<0.25):
					
					star_mag.append(float(sky_u_psf[i]))
					star_magerr.append(float(sky_u_psf_err[i]))
					star_ra.append(float(sky_ra[i]))
					star_dec.append(float(sky_dec[i]))
		
		if waveband == 'g':
			if (sky_g_psf[i] != '' and sky_g_petro[i] != ''):
				if (np.abs(float(sky_g_psf[i]) - float(sky_g_petro[i]))<0.25):
					
					star_mag.append(float(sky_g_psf[i]))
					star_magerr.append(float(sky_g_psf_err[i]))
					star_ra.append(float(sky_ra[i]))
					star_dec.append(float(sky_dec[i]))

		if waveband == 'r':
			if (sky_r_psf[i] != '' and sky_r_petro[i] != ''):
				if (np.abs(float(sky_r_psf[i]) - float(sky_r_petro[i]))<0.25):
					
					star_mag.append(float(sky_r_psf[i]))
					star_magerr.append(float(sky_r_psf_err[i]))
					star_ra.append(float(sky_ra[i]))
					star_dec.append(float(sky_dec[i]))
				
		if waveband == 'i' :           
			if (sky_i_psf[i] != '' and sky_i_petro[i] != ''):
				if (np.abs(float(sky_i_psf[i]) - float(sky_i_petro[i]))<0.25):
					
					star_mag.append(float(sky_i_psf[i]))
					star_magerr.append(float(sky_i_psf_err[i]))
					star_ra.append(float(sky_ra[i]))
					star_dec.append(float(sky_dec[i]))

		if waveband == 'z' :           
			if (sky_z_psf[i] != '' and sky_z_petro[i] != ''):
				if (np.abs(float(sky_z_psf[i]) - float(sky_z_petro[i]))<0.25):
					star_mag.append(float(sky_z_psf[i]))
					star_magerr.append(float(sky_z_psf_err[i]))
					star_ra.append(float(sky_ra[i]))
					star_dec.append(float(sky_dec[i]))
	
	# Create list with catalogue name
	star_cat = ['SkyMapper'] * len(star_ra)	
	return star_ra,star_dec,star_mag,star_magerr,star_cat


def mag_calib(directory,star_ra,star_dec,star_mag,star_magerr,star_cat,
	ra,dec,sepmag,sepmagerr,targetra,targetdec,sigma,xpixel,ypixel):	
	# Calibrate magnitude offset using field stars
	
	
	def calib_plot(xdata,ydata,xerrors,yerrors,sigma,figname):    
		# Find calibration offset
		
		# Set up figure
		fig = plt.figure()
		ax1 = fig.add_subplot(211)
		ax1.set_ylabel('Real Mag (AB)')
		ax1.invert_xaxis()
		ax1.invert_yaxis()
		ax1.errorbar(x=xdata,y=ydata,xerr=xerrors,yerr=yerrors,fmt='.')

		### Just fitting a straight line for offset
		def line1(x,c):
			return x + c

		def line2(x,c):
			return 0*x + c

		param, pcov = curve_fit(line1, xdata, ydata, sigma=yerrors)
		paramerr = np.sqrt(np.diag(pcov))

		### Plot real mag vs instrumental mag, 1-sigma errors and clipped data
		### boundaries (when appropriate)
		xx = np.linspace(min(xdata),max(xdata),1000)
		ymod1 = line1(xx,param)
		ymodhighl = line1(xx,param+paramerr)
		ymodlowl = line1(xx,param-paramerr)
		ax1.plot(xx,ymod1,color='red')
		ax1.plot(xx,ymodhighl,color='red',linestyle='--')
		ax1.plot(xx,ymodlowl,color='red',linestyle='--')
		
		if figname == 'calib.png':
			ymodhigh1c = line1(xx,param+sigma*paramerr)
			ymodlow1c = line1(xx,param-sigma*paramerr)
			ax1.plot(xx,ymodhigh1c,color='blue',linestyle='--')
			ax1.plot(xx,ymodlow1c,color='blue',linestyle='--')

		### Plot delta mag (offset) vs instrumental mag
		delta = []
		deltaerrors = []
		
		for i in range(len(xdata)):		
			delta.append(ydata[i] - xdata[i])
			deltaerrors.append(np.sqrt(xerrors[i]**2 + yerrors[i]**2))
			
		ymod2 = line2(xx,param)
		ymodhigh2 = line2(xx,param+paramerr)
		ymodlow2 = line2(xx,param-paramerr)
		ax2 = fig.add_subplot(212,sharex=ax1)
		ax2.errorbar(x=xdata,y=delta,xerr=xerrors,yerr=deltaerrors,fmt='.')
		ax2.plot(xx,ymod2,color='red')
		ax2.plot(xx,ymodhigh2,color='red',linestyle='--')
		ax2.plot(xx,ymodlow2,color='red',linestyle='--')
		ax2.set_ylabel('$\Delta$ Mag')
		ax2.set_xlabel('SEP Calculated Mag')
		
		if figname == 'calib.png':
			ymodhigh2c = line2(xx,param+sigma*paramerr)
			ymodlow2c = line2(xx,param-sigma*paramerr)
			ax2.plot(xx,ymodhigh2c,color='blue',linestyle='--')
			ax2.plot(xx,ymodlow2c,color='blue',linestyle='--')

		plt.setp(ax1.get_xticklabels(), visible=False)
		plt.savefig(directory+figname)		
		return param, paramerr
		
	
	def clipp(xdata,ydata,xerrors,yerrors,param,paramerr,sigma):
		# Clips any data that is X-sigma or further away from the best
		# fitting line
		
		xdatac = []
		xerrorc = []
		ydatac = []
		yerrorc = []
		
		# Clip data between median and X-sigma limits
		for i in range(len(xdata)):	
			yhigh = xdata[i]+float(param)+sigma*float(paramerr)
			ylow = xdata[i]+float(param)-sigma*float(paramerr)
			
			if (ylow < ydata[i] < yhigh): 
				ydatac.append(ydata[i])
				yerrorc.append(yerrors[i])
				xdatac.append(xdata[i])
				xerrorc.append(xerrors[i])
				
			else:
				continue
				
		return xdatac, xerrorc, ydatac, yerrorc
	

	### For catalogue star mags with given error as 0, change this to 0.1 mag
	star_magerr = [0.1 if (x == 0.0 or x == -0.0) else x for x in star_magerr]

	### Crossmatch SExtractor detections output with nearby known stars
	### for calibration and print to file
	cm_error_region = 1
	detect_ra = []
	detect_dec = []
	real_mag = []
	real_mage = []
	fake_mag = []
	fake_mage = []

	orig_stdout = sys.stdout
	resultf = open('cm_results.txt', "w")
	sys.stdout = resultf
	print('# DetRA, ','DetDec, ','CatRA, ','CatDec, ','MagCat, ','MagCatErr,',
		'MagIn, ','MagInErr, ','Catalogue')
	
	for i in range(len(sepmag)):	
		targetra_diff = (ra[i] - targetra)/0.000277778
		targetdec_diff = (dec[i] - targetdec)/0.000277778
		targetdiff = (targetra_diff**2 + targetdec_diff**2)**0.5
		
		if targetdiff > 5:
		
			for j in range(len(star_ra)):			
				ra_diff = (ra[i] - star_ra[j])/0.000277778
				dec_diff = (dec[i] - star_dec[j])/0.000277778
				dist = (ra_diff**2 + dec_diff**2)**0.5
				
				if dist <= cm_error_region:
				
					if (ra[i] in detect_ra and dec[i] in detect_dec):
						continue
						
					else:
						detect_ra.append(ra[i])
						detect_dec.append(dec[i])
						real_mag.append(star_mag[j])
						real_mage.append(star_magerr[j])
						fake_mag.append(sepmag[i])
						fake_mage.append(sepmagerr[i])
						print(round(ra[i],5),',',round(dec[i],5),',',
							round(star_ra[j],5),',',round(star_dec[j],5),',',
							round(star_mag[j],5),',',round(star_magerr[j],5),
							',',round(sepmag[i],5),',',round(sepmagerr[i],5),
							',',star_cat[j])

	sys.stdout = orig_stdout
	resultf.close()

	### Exit Programme if too few field stars to calibrate offset
	if len(real_mag) < 3:
		print('Not enough field stars to calibrate offset')
		sys.exit()
		
	### Plot raw data and set up clipping tool
	plot = calib_plot(fake_mag,real_mag,fake_mage,real_mage,sigma,'calib.png')
	param = plot[0]
	paramerr = plot[1]
	print('Magnitude Offset (no clipping) =',param,'+-',paramerr)
		
	### Plot clipped data and print offsets
	one = clipp(fake_mag,real_mag,fake_mage,real_mage,param,paramerr,sigma) 
	fake_magc = one[0]
	fake_magec = one[1]
	real_magc = one[2]
	real_magec = one[3]
	
	if (len(fake_magc) == len(fake_mag) or len(fake_magc) < 3):
		mag_f = np.array(sepmag) + param
		magerr_f = np.sqrt(np.array(sepmagerr)**2+paramerr**2)
		print('Fit cannot be further improved by clipping!')
		
	else:
		pc = calib_plot(fake_magc,real_magc,fake_magec,real_magec,
			sigma,'calib_c.png')
		paramc = pc[0]
		paramerrc = pc[1]
		mag_f = np.array(sepmag) + paramc
		magerr_f = np.sqrt(np.array(sepmagerr)**2+paramerrc**2)
		print('Magnitude Offset (clipped) =',paramc,'+-',paramerrc)
	
	### Print calibrated source results to text file
	orig_stdout = sys.stdout
	resultf = open('calibrated_sources.txt', "w")
	sys.stdout = resultf
	print("# mag, magerr, xpixel, ypixel, ra, dec")	
	
	for i in range(len(mag_f)):
		print(round(mag_f[i],5),',',round(magerr_f[i],5),',',round(xpixel[i],
			5),',',round(ypixel[i],5),',',round(ra[i],5),',',round(dec[i],5))
			
	sys.stdout = orig_stdout
	resultf.close()

	### Close matplotlib windows
	plt.close('all')
	return 0
	
	
def main():
	# Run the script if used from command line
	directory,gain,searchrad,waveband,im_file,sigma,aperture = get_args()
	
	# Perform photometry on image
	photresults = im_phot(directory,gain,im_file,aperture)
	targetra,targetdec,sepmag,sepmagerr,ra,dec,xpixel,ypixel = photresults
	
	# Scrape the four catalogues for field stars
	sdss_ra,sdss_dec,sdss_mag,sdss_magerr,sdss_cat = sdss_search(searchrad,
		waveband,targetra,targetdec)
	ps_ra,ps_dec,ps_mag,ps_magerr,ps_cat = panstarrs_search(searchrad,
		waveband,targetra,targetdec)
	apass_ra,apass_dec,apass_mag,apass_magerr,apass_cat = apass_search(
		searchrad,waveband,targetra,targetdec)
	sm_ra,sm_dec,sm_mag,sm_magerr,sm_cat = skymapper_search(searchrad,
		waveband,targetra,targetdec)
		
	# Add all above star properties to single lists and find mag offset
	star_ra = sdss_ra + ps_ra + apass_ra + sm_ra
	star_dec = sdss_dec + ps_dec + apass_dec + sm_dec
	star_mag = sdss_mag + ps_mag + apass_mag + sm_mag
	star_magerr = sdss_magerr + ps_magerr + apass_magerr + sm_magerr
	star_cat = sdss_cat + ps_cat + apass_cat + sm_cat
	
	return mag_calib(directory,star_ra,star_dec,star_mag,star_magerr,star_cat,
		ra,dec,sepmag,sepmagerr,targetra,targetdec,sigma,xpixel,ypixel)
		

if __name__ == '__main__':
    sys.exit(main())