#!/usr/bin/python
# -*- coding: utf-8 -*-
# FORS1 polarisation photometry v1.0
# Created by Adam Higgins
# Email: abh13@le.ac.uk

__doc__ = """ Script runs photometry for the polarisation images from FORS2
and outputs flux information for ordinary and extraordinary beams. This
script assumes that the ordinary beam is the top one!

File names for each half-wave plate angle should be:
0ang.fits,
225ang.fits,
45ang.fits,
675ang.fits
"""

from astropy.io import fits
from astropy.table import vstack
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch
from astropy.visualization import ZScaleInterval
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import DAOStarFinder
from photutils import CircularAperture
from photutils import aperture_photometry
from photutils import RectangularAnnulus
from photutils.utils import calc_total_error
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse


def get_args():
	""" Parse command line arguments """
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("Directory",metavar="DIR",type=str,action="store",
		help="Required directory")
	parser.add_argument("--ap",type=float,default=2.0,dest='aperture',
		help="Source aperture diameter X*FWHM (default = 2.0)")
		
	args = parser.parse_args()
	folder_path = args.__dict__['Directory']
	apermul = args.aperture
	return folder_path,apermul


def fors2_pol_phot(folder_path,apermul):
	""" Script runs photometry for FORS2 polarimetry images """

	# Read in four wave plate angle files and set up arrays for later
	file_ang0 = os.path.join(folder_path,'0ang.fits')
	file_ang225 = os.path.join(folder_path,'225ang.fits')
	file_ang45 = os.path.join(folder_path,'45ang.fits')
	file_ang675 = os.path.join(folder_path,'675ang.fits')

	files = [file_ang0,file_ang225,file_ang45,file_ang675]
	angle = ['0','225','45','675']
	ang_dec = ['0','22.5','45','67.5']
	label = ['$0^{\circ}$ image','$22.5^{\circ}$ image',
		'$45^{\circ}$ image','$67.5^{\circ}$ image']
		
	# Set up array to store FWHM and number of sources per half-wave plate
	fwhm = []
	numsource = []
	
	# Loop over files for the four wave plate files
	for k in range(0,len(angle),1):

		# Open fits file, extract pixel flux data and remove saturated pixels
		try:
			hdulist = fits.open(files[k])
			image_data = hdulist[0].data
			
		except FileNotFoundError as e:
			print("Cannot find the fits file(s) you are looking for!")
			print("Please check the input!")
			sys.exit()
		
		# Remove bad pixels and mask edges
		image_data[image_data > 60000] = 0
		image_data[image_data < 0] = 0    
		rows = len(image_data[:,0])
		cols = len(image_data[0,:])
		hdulist.close()

		# Calculate estimate of background using sigma-clipping and detect
		# the sources using DAOStarFinder
		xord, yord = 843, 164
		xexord, yexord = 843, 72
		go_bmean, go_bmedian, go_bstd = sigma_clipped_stats(image_data
			[yord-40:yord+40,xord-40:xord+40],sigma=3.0,iters=5)
		ge_bmean, ge_bmedian, ge_bstd = sigma_clipped_stats(image_data
			[yexord-40:yexord+40,xord-40:xord+40],sigma=3.0,iters=5)
		daofind_o = DAOStarFinder(fwhm=5,threshold=5*go_bstd,
			exclude_border=True)
		daofind_e = DAOStarFinder(fwhm=5,threshold=5*ge_bstd,
			exclude_border=True)
		sources_o = daofind_o(image_data[yord-20:yord+20,xord-20:xord+20])
		sources_e = daofind_e(image_data[yexord-20:yexord+20,xexord-20:
			xexord+20])
		
		if (len(sources_o) < 1 or len(sources_e) < 1):
			print("No source detected in",ang_dec[k],"degree image")
			sys.exit()
			
		if len(sources_o) != len(sources_e):
			print("Unequal number of sources detected in o and e images!")
			sys.exit()
		
		glob_bgm = [go_bmean,ge_bmean]
		glob_bgerr = [go_bstd,ge_bstd]
		
		# Convert the source centroids back into detector pixels
		sources_o['xcentroid'] = sources_o['xcentroid'] + xord - 20
		sources_o['ycentroid'] = sources_o['ycentroid'] + yord - 20
		sources_e['xcentroid'] = sources_e['xcentroid'] + xexord - 20
		sources_e['ycentroid'] = sources_e['ycentroid'] + yexord - 20

		# Estimate the FWHM of the source by simulating a 2D Gaussian
		# (only done on 0 angle image ensuring aperture sizes are equal)	
		if not fwhm:
			xpeaks_o = []
			xpeaks_e = []
			ypeaks_o = []
			ypeaks_e = []
			
			for i in range(0,len(sources_o),1):			
				data_o = image_data[yord-20:yord+20,xord-20:xord+20]
				xpeaks_o.append(int(sources_o[i]['xcentroid']) - (xord - 20))
				ypeaks_o.append(int(sources_o[i]['ycentroid']) - (yord - 20))
					
				data_e = image_data[yexord-20:yexord+20,xexord-20:xexord+20]
				xpeaks_e.append(int(sources_e[i]['xcentroid']) - 
					(xexord - 20))
				ypeaks_e.append(int(sources_e[i]['ycentroid']) - 
					(yexord - 20))
				
				min_count_o = np.min(data_o)
				min_count_e = np.min(data_e)
				max_count_o = data_o[ypeaks_o[i],xpeaks_e[i]]
				max_count_e = data_e[ypeaks_o[i],xpeaks_e[i]]
				half_max_o = (max_count_o + min_count_o)/2
				half_max_e = (max_count_e + min_count_e)/2
				
				# Crude calculation for each source
				nearest_above_x_o = ((np.abs(data_o[ypeaks_o[i],
					xpeaks_o[i]:-1] - half_max_o)).argmin())
				nearest_below_x_o = ((np.abs(data_o[ypeaks_o[i],0:
					xpeaks_o[i]] - half_max_o)).argmin())
				nearest_above_x_e = ((np.abs(data_e[ypeaks_e[i],
					xpeaks_e[i]:-1] - half_max_e)).argmin())
				nearest_below_x_e = ((np.abs(data_e[ypeaks_e[i],0:
					xpeaks_e[i]] - half_max_e)).argmin())
				nearest_above_y_o = ((np.abs(data_o[ypeaks_o[i]:-1,
					xpeaks_o[i]] - half_max_o)).argmin())
				nearest_below_y_o = ((np.abs(data_o[0:ypeaks_o[i],
					xpeaks_o[i]] - half_max_o)).argmin())
				nearest_above_y_e = ((np.abs(data_e[ypeaks_e[i]:-1,
					xpeaks_e[i]] - half_max_e)).argmin())
				nearest_below_y_e = ((np.abs(data_e[0:ypeaks_e[i],
					xpeaks_e[i]] - half_max_e)).argmin())
				fwhm.append((nearest_above_x_o + (xpeaks_o[i] -
					nearest_below_x_o)))
				fwhm.append((nearest_above_y_o + (ypeaks_o[i] -
					nearest_below_y_o)))
				fwhm.append((nearest_above_x_e + (xpeaks_e[i] -
					nearest_below_x_e)))
				fwhm.append((nearest_above_y_e + (ypeaks_e[i] -
					nearest_below_y_e)))
			
			fwhm = np.mean(fwhm)
		
		# Stack both ord and exord sources together
		tot_sources = vstack([sources_o,sources_e])
				
		# Store the ordinary and extraordinary beam source images and
		# create apertures for aperture photometry 
		positions = np.swapaxes(np.array((tot_sources['xcentroid'],
			tot_sources['ycentroid']),dtype='float'),0,1)
		aperture = CircularAperture(positions, r=0.5*apermul*fwhm)
		phot_table = aperture_photometry(image_data,aperture)   
					  
		# Set up arrays of ord and exord source parameters
		s_id = np.zeros([len(np.array(phot_table['id']))])
		xp = np.zeros([len(s_id)])
		yp = np.zeros([len(s_id)])
		fluxbgs = np.zeros([len(s_id)])
		mean_bg = np.zeros([len(s_id)])
		bg_err = np.zeros([len(s_id)])
		s_area = []
		ann_area = []
		
		for i in range(0,len(np.array(phot_table['id'])),1):
			s_id[i] = np.array(phot_table['id'][i])
			xpos = np.array(phot_table['xcenter'][i])
			ypos = np.array(phot_table['ycenter'][i])
			xp[i] = xpos
			yp[i] = ypos
			s_area.append(np.pi*(0.5*apermul*fwhm)**2)
			j = i%2				
			fluxbgs[i] = (phot_table['aperture_sum'][i] -
				aperture.area()*glob_bgm[j])
			mean_bg[i] = glob_bgm[j]
			bg_err[i] = glob_bgerr[j]
			ann_area.append(80*80)			
		
		# Create and save the image in z scale and overplot the ordinary and
		# extraordinary apertures and local background annuli if applicable
		fig = plt.figure()
		zscale = ZScaleInterval(image_data)
		norm = ImageNormalize(stretch=SqrtStretch(),interval=zscale)
		image = plt.imshow(image_data,cmap='gray',origin='lower',norm=norm)
		bg_annulus_o = RectangularAnnulus((843,159),w_in=0,w_out=80,h_out=80,
			theta=0)
		bg_annulus_e = RectangularAnnulus((843,69),w_in=0,w_out=80,h_out=80,
			theta=0)
		bg_annulus_o.plot(color='skyblue',lw=1.5,alpha=0.5)
		bg_annulus_e.plot(color='lightgreen',lw=1.5,alpha=0.5)
		
		for i in range(0,len(np.array(phot_table['id'])),1):
			aperture = CircularAperture((xp[i],yp[i]),r=0.5*apermul*fwhm)
			
			if i < int(len(np.array(phot_table['id']))/2):
				aperture.plot(color='blue',lw=1.5,alpha=0.5)
		
			else:
				aperture.plot(color='green',lw=1.5,alpha=0.5)
			
		plt.xlim(760,920)
		plt.ylim(20,210)
		plt.title(label[k])
		image_fn = folder_path + angle[k] + '_image.png'
		fig.savefig(image_fn)

		# Write ordinary and extraordinary beams to file following the 
		# convention angleXXX_ord.txt and angleXXX_exord.txt
		orig_stdout = sys.stdout
		ord_result_file= folder_path + 'angle' + angle[k] + '_ord.txt'
		ordresultf = open(ord_result_file, 'w')
		sys.stdout = ordresultf
		
		print("# id, xpix, ypix, fluxbgs, sourcearea, meanbg, bgerr, bgarea") 
		for i in range(0,int(len(np.array(phot_table['id']))/2),1):
			print(i+1,xp[i],yp[i],fluxbgs[i],s_area[i],mean_bg[i],bg_err[i],
				ann_area[i])
		sys.stdout = orig_stdout
		ordresultf.close()

		orig_stdout = sys.stdout
		exord_result_file = folder_path + 'angle' + angle[k] + '_exord.txt'
		exordresultf = open(exord_result_file, 'w')
		sys.stdout = exordresultf
		
		print("# id, xpix, ypix, fluxbgs, sourcearea, meanbg, bgerr, bgarea")
		for i in range(int(len(np.array(phot_table['id']))/2),len(np.array
			(phot_table['id'])),1):
			print(i+1-int(len(np.array(phot_table['id']))/2),xp[i],yp[i],
				fluxbgs[i],s_area[i],mean_bg[i],bg_err[i],ann_area[i])  
		sys.stdout = orig_stdout
		exordresultf.close()
		
		# Save the number of sources in each beam to a list
		numsource.append(int(len(np.array(phot_table['id']))/2))
	
	# Print number of sources per half-wave plate image
	for i in range(0,len(numsource),1):
		print("No of sources detected at",ang_dec[i],"degrees:",numsource[i])
	
	return 0

	
def main():
	""" Run script from command line """
	folder_path,apermul = get_args()
	return fors2_pol_phot(folder_path,apermul)

	
if __name__ == '__main__':
    sys.exit(main())