#!/usr/bin/python/
# -*- coding: utf-8 -*-
# Created by Adam Higgins

__doc__ = """ FORS2 script for imaging and polarimetry raw file reduction
for observations using the 2x2 binning.

The script can perform three tasks - (1) combining bias frames to one master
bias frame, (2) combining flat frames which have been bias corrected to one
master flat frame and (3) subtracting these two master frames from the raw
input image where the output is saved as FB_XXXX.fits where XXXX.fits is the
raw input file name.

All output files are reshaped from the initial 2048x1034 data array to
1685x940 to remove the top/right edges of the CCD which are noisy and can
affect the clipping.

NOTE: The script only reduces using imaging flats - this is usually fine
as FORS2 doesn't rotate the wavelplate for flats taken with polarisation
optics in the beam 
"""

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
import os
import glob
import sys
import argparse


def get_args():
	""" Parse command line arguments """
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("Directory",metavar="DIR",type=str,action='store',
		help="Required directory")
	parser.add_argument("Task",metavar="TASK",type=str,action="store",
		help="Required task - BIAS/FLAT/RED")
	parser.add_argument("Files",nargs='*',help="Parse up to three file names\
		dependent on task - one each for MBIAS/MFLAT/RAWIMAGE")
	parser.add_argument("--ft",type=str,default='IMSKY',dest="flattype",
		help="Flat type - IMSKY/IMDOME (default = IMSKY)")
	
	# Return correct amount of arguments for task
	args = parser.parse_args()
	task = args.__dict__['Task']
	files = args.__dict__['Files']
	directory = args.__dict__['Directory']
	flattype = args.flattype
	
	if len(files) == 1 and task == 'BIAS':
		masterb = files[0]
		return directory,task,flattype,masterb
		
	if len(files) == 2 and task == 'FLAT':
		masterb, masterf = files
		return directory,task,flattype,masterb,masterf
		
	if len(files) == 3 and task == 'RED':
		masterb, masterf, imfile = files	
		return directory,task,flattype,masterb,masterf,imfile
		
	else:
		print("Error in arguments. Check number of arguments for task!")
		sys.exit()
		

def master_bf(masterb):
	""" Produces a master bias fits file """
	bias_fn = []
	
	for file in glob.glob('*.fits'):
		hdulist = fits.open(file)
		header = hdulist[0].header
		
		if (header['object'] == 'BIAS' and header['EXTNAME'] == 'CHIP1' and
			header['NAXIS1'] == 2048 and header['NAXIS2'] == 1034):
			bias_fn.append(file)
			
		hdulist.close()
	
	bstack_data = np.zeros([len(bias_fn),940,1665])
	i = 0
	
	for file in bias_fn:
		hdulist = fits.open(file)
		data = hdulist[0].data
		header = hdulist[0].header
		
		bstack_data[i,:,:] = data[6:946,190:1855]
		i += 1
			
		hdulist.close()
	
	print("Number of applicable bias frames used in master:",i)
	
	if i > 0:
		bstack_mean = np.nanmean(bstack_data,axis=0)
		bstack_std = np.nanstd(bstack_data,axis=0)

		hdu = fits.PrimaryHDU(bstack_mean)
		hdu.header['object'] = 'BIAS'
		hdu.header['frames'] = len(bias_fn)
		hdu.header['mean'] = np.nanmean(bstack_mean)
		hdu.header['std'] = np.nanstd(bstack_std)
		hdu.header['naxis1'] = 1665
		hdu.header['naxis2'] = 940
		hdu.header['binning'] = 2

		hdulist = fits.HDUList([hdu])
		hdulist.writeto(masterb)
		return 0
	
	else:
		print("No applicable bias frames!")
		sys.exit()

	
def master_ff(flattype,masterb,masterf):
	""" Produces a master flat that has been bias corrected """
	flat_fn = []
	
	for file in glob.glob('*.fits'):
		hdulist = fits.open(file)
		header = hdulist[0].header
		
		if flattype == 'IMSKY':
		
			if (header['object'] == 'FLAT,SKY' and header['EXTNAME'] ==
				'CHIP1'	and	header['NAXIS1'] == 2048 and header['NAXIS2']
				== 1034 and	header['HIERARCH ESO DPR TECH'] == 'IMAGE'):
				flat_fn.append(file)
				
		if flattype == 'IMDOME':
			
			if (header['object'] == 'FLAT,LAMP' and header['EXTNAME'] ==
				'CHIP1'	and	header['NAXIS1'] == 2048 and header['NAXIS2']
				== 1034 and	header['HIERARCH ESO DPR TECH'] == 'IMAGE'):
				flat_fn.append(file)
			
		hdulist.close()
		
	fstack_data = np.zeros([len(flat_fn),940,1665])
	mblist = fits.open(masterb)
	mbdata = mblist[0].data
	mblist.close()
	i = 0
	
	for file in flat_fn:
		hdulist = fits.open(file)
		data = hdulist[0].data
		header = hdulist[0].header
		
		bc_data = data[6:946,190:1855] - mbdata
		clipped_data = sigma_clip(bc_data,sigma_upper=3,sigma_lower=104)
		bc_data[clipped_data.mask == True] = np.nan
		fstack_data[i,:,:] = bc_data
		i += 1
			
		hdulist.close()	
		
	print("Number of applicable flat frames used in master:",i)
	
	if i > 0:
		fstack_sum = np.nanmean(fstack_data,axis=0)
		fmedian = np.nanmedian(fstack_sum)
		fstack_sum[np.isnan(fstack_sum)] = fmedian
		flat_array = fstack_sum/fmedian
		fstd = np.nanstd(flat_array)

		hdu = fits.PrimaryHDU(flat_array)
		
		if flattype == 'IMSKY':
			hdu.header['object'] = 'MASTER FLAT,SKY'
			
		if flattype == 'IMDOME':
			hdu.header['object'] = 'MASTER FLAT,LAMP'
			
		hdu.header['obstech'] = 'IMAGE'
		hdu.header['frames'] = len(flat_fn)
		hdu.header['median'] = 1
		hdu.header['std'] = fstd
		hdu.header['naxis1'] = 1665
		hdu.header['naxis2'] = 940
		hdu.header['binning'] = 2

		hdulist = fits.HDUList([hdu])
		hdulist.writeto(masterf)
		return 0
		
	else:
		print("No applicable flat frames")
		sys.exit()
		
		
def reduce_file(masterb,masterf,imfile):
	""" Reduces the raw science image, subtracting the flats and bias frames
	either created with this script or existing ones that have been read in.
	"""
	
	mblist = fits.open(masterb)
	mbdata = mblist[0].data
	mblist.close()
	mflist = fits.open(masterf)
	mfdata = mflist[0].data
	mflist.close()
	
	imlist = fits.open(imfile)
	imheader = imlist[0].header
	imdata = imlist[0].data
	imlist.close()
	
	if np.shape(imdata) == (1034,2048):
		imdata_b = imdata[6:946,190:1855] - mbdata
		imdata_fb = imdata_b/mfdata
		hdu = fits.PrimaryHDU(imdata_fb,header=imheader)
		hdulist= fits.HDUList([hdu])
		newfile = 'FB_' + imfile
		hdulist.writeto(newfile,overwrite=True,output_verify='ignore')
		return 0
		
	else:
		print("Raw image data is not of initial shape 2048x1034")
		sys.exit()
		
	
def main():
	""" Run script from command line """
	
	# Deal with bias frames
	args = get_args()
	directory = args[0]
	task = args[1]
	flattype = args[2]
	masterb = args[3]
	os.chdir(directory)
	
	if os.path.isfile(masterb) != True:
		make_mbf = master_bf(masterb)
		
	else:
		print("Master bias already exists - using existing file")
			
	# Deal with flat frames	
	if len(args) >= 5:
		masterf = args[4]
		
		if os.path.isfile(masterf) != True:
		
			make_mff = master_ff(flattype,masterb,masterf)
				
		else:
			print("Master flat already exists - using existing file")
				
	# Deal with image reduction
	if len(args) == 6:
		imfile = args[5]
		
		if os.path.isfile(imfile) == True:
			imred = reduce_file(masterb,masterf,imfile)
			print('done')
			
		else:
			print("Can't find raw image file, please check input")		
		
	return 0
	
	
if __name__ == '__main__':
    sys.exit(main())