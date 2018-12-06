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
675ang.fits """

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
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

def get_args():
	# Parse command line arguments
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("Directory",metavar="DIR",type=str,action="store",
		help='Required directory')
	parser.add_argument("Background",metavar="BKG",type=str,action="store",
		help="Background calculation to use for photometry calibration - \
		choices are GLOBAL/LOCAL")
	parser.add_argument("--ap",type=float,default=1.5,dest='aperture',
		help="Source aperture diameter size X*FWHM (default = 1.5)")
		
	args = parser.parse_args()
	
	directory = args.__dict__['Directory']
	bkg_type = args.__dict__['Background']
	apermul = args.aperture
	return directory,bkg_type,apermul
	
	
def fors2_pol_phot(directory,bkg_type,apermul):
	""" Perform photometry on the four wave-plate angle images """
	
	# Make sure background type is valid!
	if (bkg_type != 'GLOBAL' and bkg_type != 'LOCAL'):
		print('Please choose a valid background estimation')
		sys.exit()

	# Read in four wave plate angle files and set up arrays for later
	file_ang0 = os.path.join(directory,'0ang.fits')
	file_ang225 = os.path.join(directory,'225ang.fits')
	file_ang45 = os.path.join(directory,'45ang.fits')
	file_ang675 = os.path.join(directory,'675ang.fits')

	files = [file_ang0,file_ang225,file_ang45,file_ang675]
	angle = ['0','225','45','675']
	ang_dec = ['0','22.5','45','67.5']
	label = ['$0^{\circ}$ image','$22.5^{\circ}$ image',
			 '$45^{\circ}$ image','$67.5^{\circ}$ image']

	for k in range(0,len(angle),1):

		# Open fits file, extract pixel flux data and remove saturated pixels
		try:
			hdulist = fits.open(files[k])
			image_data = hdulist[0].data
			
		except FileNotFoundError as e:
			print('Cannot find the fits file(s) you are looking for!')
			print('Please check the input!')
			sys.exit()
		

		# Remove bad pixels and mask edges
		image_data[image_data > 60000] = 0
		image_data[image_data < 0] = 0    
		rows = len(image_data[:,0])
		cols = len(image_data[0,:])
		hdulist.close()

		# Calculate rough estimate of background and detect the target source
		# using DAOStarFinder
		xord, yord = 843, 164
		xexord, yexord = 843, 72
		o_bmean, o_bmedian, o_bstd = sigma_clipped_stats(image_data
			[yord-40:yord+30,:],sigma=3.0,iters=5)
		e_bmean, e_bmedian, e_bstd = sigma_clipped_stats(image_data
			[yexord-38:yexord+32,:],sigma=3.0,iters=5)
		daofind_o = DAOStarFinder(fwhm=5,threshold=5*o_bstd,exclude_border=True)
		daofind_e = DAOStarFinder(fwhm=5,threshold=5*e_bstd,exclude_border=True)
		sources_o = daofind_o(image_data[yord-18:yord+18,xord-18:xord+18])
		sources_e = daofind_e(image_data[yexord-18:yexord+18,xexord-18:xexord+18])
		
		if (len(sources_o) < 1 or len(sources_e) < 1):
			print('No source detected in',ang_dec[k],'degree image')
			sys.exit()
			
		glob_bgm = [o_bmean,e_bmean]
		glob_bgerr = [o_bstd,e_bstd]
		
		# Convert the source centroids back into detector pixels
		sources_o['xcentroid'] = sources_o['xcentroid'] + xord - 18
		sources_o['ycentroid'] = sources_o['ycentroid'] + yord - 18
		sources_e['xcentroid'] = sources_e['xcentroid'] + xexord - 18
		sources_e['ycentroid'] = sources_e['ycentroid'] + yexord - 18

		# Estimate the FWHM of the source by simulating a 2D Gaussian (crudely)
		fwhm = []
		
		for i in range(0,2,1):
		
			if i == 0:
				data_c = image_data[yord-15:yord+15,xord-15:xord+15]
				xpeak = int(sources_o['xcentroid']) - (xord - 15)
				ypeak = int(sources_o['ycentroid']) - (yord - 15)
				
			if i == 1:
				data_c = image_data[yexord-15:yexord+15,xexord-15:xexord+15]
				xpeak = int(sources_e['xcentroid']) - (xexord - 15)
				ypeak = int(sources_e['ycentroid']) - (yexord - 15)
				
			min_count = np.min(data_c)
			max_count = np.max(data_c)
			half_max = (max_count + min_count)/2
			nearest_above_x = (np.abs(data_c[ypeak,xpeak:-1]-half_max)).argmin()
			nearest_below_x = (np.abs(data_c[ypeak,0:xpeak]-half_max)).argmin()
			nearest_above_y = (np.abs(data_c[ypeak:-1,xpeak]-half_max)).argmin()
			nearest_below_y = (np.abs(data_c[0:ypeak,xpeak]-half_max)).argmin()
			fwhm.append((nearest_above_x + (xpeak - nearest_below_x)))
			fwhm.append((nearest_above_y + (ypeak - nearest_below_y)))
			
		fwhm = np.mean(fwhm)
		
		# Stack both ord and exord sources together
		tot_sources = vstack([sources_o,sources_e])
				
		# Store the ordinary and extraordinary beam source images and
		# create apertures for aperture photometry 
		ann_in = 4*fwhm
		ann_out = ann_in + 10
		positions = ((tot_sources['xcentroid'][0],tot_sources['ycentroid']
			[0]),(tot_sources['xcentroid'][1],tot_sources['ycentroid'][1]))
		aperture = CircularAperture(positions, r=0.5*apermul*fwhm)
		phot_table = aperture_photometry(image_data,aperture)
		
		def local_back_stats(position):
			# Create custom rectangular background annulus for estimation of
			# local background of mean and standard error when applicable
			
			xpixel, ypixel = position
			xlowi = int(round(xpixel - ann_in/2))
			xlowo = int(round(xpixel - ann_out/2))
			xhii = int(round(xpixel + ann_in/2))
			xhio = int(round(xpixel + ann_in/2))
			ylowi = int(round(ypixel - ann_in/2))
			ylowo = int(round(ypixel - ann_out/2))
			yhii = int(round(ypixel + ann_in/2))
			yhio = int(round(ypixel + ann_in/2))
			bg_data = []
			
			for i in range(xlowo,xlowi+1,1):			
				for j in range(ylowo,yhio+1,1):
					bg_data.append(image_data[j,i]) 
					
			for i in range(xlowi,xhii+1,1):			
				for j in range(ylowo,ylowi+1,1):
					bg_data.append(image_data[j,i])
					
			for i in range(xlowi,xhii+1,1):			
				for j in range(yhii,yhio+1,1):
					bg_data.append(image_data[j,i])
					
			for i in range(xhii,xhio+1,1):			
				for j in range(ylowo,yhio+1,1):
					bg_data.append(image_data[j,i])
					
			bg_data = np.array(bg_data)
			ann_area = len(bg_data)
			source_ap = CircularAperture((position),r=0.5*apermul*fwhm)
			tot_source_bg = np.mean(bg_data)*source_ap.area()
			return tot_source_bg, np.mean(bg_data), np.std(bg_data), ann_area   
					  
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
			
			if bkg_type == 'LOCAL':
				local_back = local_back_stats([xpos,ypos])
				fluxbgs[i] = phot_table['aperture_sum'][i] - local_back[0]
				mean_bg[i] = local_back[1]
				bg_err[i] = local_back[2]
				ann_area.append(local_back[3])
				
			if bkg_type == 'GLOBAL':
				fluxbgs[i] = (phot_table['aperture_sum'][i] -
							  aperture.area()*glob_bgm[i])
				mean_bg[i] = glob_bgm[i]
				bg_err[i] = glob_bgerr[i]
				ann_area.append(aperture.area())			

		# Create and save the image in z scale and overplot the ordinary and
		# extraordinary apertures and local background annuli if applicable
		fig = plt.figure()    
		aperture_ord = CircularAperture((xp[0],yp[0]),r=0.5*apermul*fwhm)
		aperture_exord = CircularAperture((xp[1],yp[1]),r=0.5*apermul*fwhm)
		bg_annulus_ord = RectangularAnnulus((xp[0],yp[0]),w_in=ann_in,
			w_out=ann_out,h_out=ann_out,theta=0)
		bg_annulus_exord = RectangularAnnulus((xp[1],yp[1]),w_in=ann_in,
			w_out=ann_out,h_out=ann_out,theta=0)
		zscale = ZScaleInterval(image_data)
		norm = ImageNormalize(stretch=SqrtStretch(),interval=zscale)
		image = plt.imshow(image_data,cmap='gray',origin='lower',norm=norm)
		aperture_ord.plot(color='blue', lw=1.5, alpha=0.5)
		aperture_exord.plot(color='green', lw=1.5, alpha=0.5)
		
		if bkg_type == 'LOCAL':
			bg_annulus_ord.plot(color='skyblue', lw=1.5, alpha=0.5)
			bg_annulus_exord.plot(color='lightgreen', lw=1.5, alpha=0.5)
			
		plt.xlim(790,890)
		plt.ylim(50,200)
		plt.title(label[k])
		image_fn = directory + angle[k] + '_image.png'
		fig.savefig(image_fn)

		# Write ordinary and extraordinary beams to file following the convention
		# angleXXX_ord.txt and angleXXX_exord.txt
		orig_stdout = sys.stdout
		ord_result_file= directory + 'angle' + angle[k] + '_ord.txt'
		ordresultf = open(ord_result_file, 'w')
		sys.stdout = ordresultf
		print('# id, xpixel, ypixel, fluxbgs, sourcearea, meanbg, bgerr, bgarea') 
		print(1,xp[0],yp[0],fluxbgs[0],s_area[0],mean_bg[0],bg_err[0],ann_area[0])
		sys.stdout = orig_stdout
		ordresultf.close()

		orig_stdout = sys.stdout
		exord_result_file = directory + 'angle' + angle[k] + '_exord.txt'
		exordresultf = open(exord_result_file, 'w')
		sys.stdout = exordresultf
		print('# id, xpixel, ypixel, fluxbgs, sourcearea, meanbg, bgerr, bgarea')
		print(1,xp[1],yp[1],fluxbgs[1],s_area[1],mean_bg[1],bg_err[1],ann_area[1])       
		sys.stdout = orig_stdout
		exordresultf.close()
		
	return 0
		
def main():
	# Run script from command line
	directory,bkg_type,apermul = get_args()
	return fors2_pol_phot(directory,bkg_type,apermul)
	
if __name__ == '__main__':
    sys.exit(main())