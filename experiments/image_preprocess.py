"""
This module has some handy functions for preprocessing images
author: Cheng Tang, chengtang48@gmail.com, June 2016
"""
import numpy as np
import random

def rsubsample(X, n_per_image, width, height, labels = None):
	"""
	 randomly subsampling patches from a set of images
	 ----------------
	 Input
	 	X: ndarray of dim 3(grey-scale) or 4(rgb), each "row" an image
	 	n_per_image: number of patches sampled per image
	 	width: width of each patch
	 	height: height of each patch
	 	
	 Output
	 	Y: ndarray of shape n_per_image * nrows(X) by width*height*color_levels
	 	Y_lb: labels of corresponding new patches
	"""
	if len(X.shape) == 3:
		## grey-scale image
		Y = np.zeros((n_per_image*X.shape[0], width*height))
		Y_lb = None
		if labels != None:
			Y_lb = np.zeros((Y.shape[0],1)) 
		yind = 0
		im_ind = 0
		for image in X:
			old_width = image.shape[0]
			old_height = image.shape[1]
			assert old_width >= width and old_height >= height, 'patch size too large!'
			
			for i in range(n_per_image):
				rind = random.randrange(0,old_width-width+1)
				cind = random.randrange(0,old_height-height+1)
				patch = image[rind:rind+width, cind:cind+height]
				Y[yind] = patch.reshape((1,width*height))
				if Y_lb != None:
					Y_lb[yind] = labels[im_ind]
				
				yind += 1
			
			im_ind += 1
		
	else:
		## rgb image
		Y = np.zeros((n_per_image*X.shape[0], width*height*3))
		Y_lb = None
		if labels != None:
			Y_lb = np.zeros((Y.shape[0],1)) 
		yind = 0
		im_ind = 0
		for image in X:
			old_width = image.shape[0]
			old_height = image.shape[1]
			assert old_width >= width and old_height >= height, 'patch size too large!'
			
			for i in range(n_per_image):
				rind = random.randrange(0,old_width-width+1)
				cind = random.randrange(0,old_height-height+1)
				patch = image[rind:rind+width, cind:cind+height,:]
				patch = patch.reshape((width*height,3))
				Y[yind] = patch.flatten() # concatenate rgb channels
				if Y_lb != None:
					Y_lb[yind] = labels[im_ind]
				
				yind += 1
			
			im_ind += 1
		
	return Y, Y_lb

def convsubsample(X, step, width, height, labels = None):
	"""
	 subsampling the image via valid convolution (no padding) 
	 ----------------
	 Input
	 	X: ndarray of dim 3(grey-scale) or 4(rgb), each "row" an image
	 	width: width of each filter
	 	height: height of each filter
	 	
	 Output
	 	Y: ndarray of shape (width(X)-width)/step+1 by (length(X)-length)/step+1
	 	Y_lb: labels of corresponding new patches
	"""
	if len(X.shape) == 3:
		## grey-scale image
		n_per_image = ((X.shape[1]-height)/step+1)*((X.shape[2]-width)/step+1) 
		Y = np.zeros((X.shape[0]*n_per_image, width*height))
		print 'size of preprocessed data will be %d by %d' %(Y.shape[0], Y.shape[1])
		Y_lb = None
		if not labels is None:
			Y_lb = np.zeros((Y.shape[0],1)) 
		yind = 0
		im_ind = 0
		for image in X:
			if im_ind % 500 == 0:
				print im_ind
				print 'processing data from %d to %d...' %(im_ind, im_ind+500) 
			old_height = image.shape[0]
			old_width = image.shape[1]
			assert old_width >= width and old_height >= height, 'patch size too large!'
			
			rind = 0
			cind = 0
			for i in xrange((X.shape[2]-width)/step+1):
				# sweeping through columns, fixing row
				
				for j in xrange((X.shape[1]-height)/step+1):
					# sweeping through rows, fixing column
					patch = image[rind:rind+height, cind:cind+width]
					Y[yind] = patch.reshape((1,width*height))
					
					if not Y_lb is None:
						Y_lb[yind] = labels[im_ind]
					
					if j < (X.shape[1]-height)/step+1:
						rind += step
					yind += 1
					
					print j,i
				
				if i < (X.shape[2]-width)/step+1:
					cind += step
				
			
			im_ind += 1
		
	else:
		## rgb image
		n_per_image = ((X.shape[1]-height)/step+1)*((X.shape[2]-width)/step+1) 
		Y = np.zeros((n_per_image*X.shape[0], width*height*3))
		print 'size of preprocessed data will be %d by %d' %(Y.shape[0], Y.shape[1])
		Y_lb = None
		if not labels is None:
			Y_lb = np.zeros((Y.shape[0],1)) 
		yind = 0
		im_ind = 0
		for image in X:
			if im_ind % 500 == 0:
				print 'processing data from %d to %d...' %(im_ind, im_ind+499) 
			old_height = image.shape[0]
			old_width = image.shape[1]
			assert old_width >= width and old_height >= height, 'patch size too large!'
			
			rind = 0
			cind = 0
			for i in xrange((X.shape[2]-width)/step+1):
				
				for j in xrange((X.shape[1]-height)/step+1):
					patch = image[rind:rind+height, cind:cind+width,:]
					patch = patch.reshape((width*height,3))
					Y[yind] = patch.flatten() # concatenate rgb channels
					
					if not (Y_lb is None):
						#print Y_lb[yind],labels[im_ind]
						Y_lb[yind] = labels[im_ind]
					
					yind += 1
				
					#print width, height, patch.shape[0], patch.shape[1]
				
					#if j < (X.shape[1]-height)/step+1:
					rind += step
			
				#if i < (X.shape[2]-width)/step+1:
				cind += step
				rind = 0
				
				#print rind,cind
				
			im_ind += 1
	
	return Y, Y_lb
	


def white(X):
	"""
	 Input
	 	X: n_data by n_features
	 Output
	 	Y: n_data by n_features; whitened data
	"""
	return Y
### Tests
if __name__ == '__main__':
	toy = np.array([1,-2,3,-4,5,-6,7,-8,1,-2,3,-4,5,-6,7,-8]).reshape([2,4,2])
	toy_rgb = np.concatenate((toy[...,np.newaxis],-toy[...,np.newaxis],toy[...,np.newaxis]),axis = 3)
	#print toy_rgb
	#Y = rsubsample(toy_rgb, 5, 2, 2)
	Y = convsubsample(toy_rgb, 1, 2, 2)
	print Y
	
	