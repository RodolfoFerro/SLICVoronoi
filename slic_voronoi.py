#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, os, cv2, math, time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse

# DEFINE:
m    = 0.
path = "imgs/"

#----------------------------------------------------------

#Functions: 
def cvec_k(grad,img,x,y,N):
	# Return the center vectors of regions:
	cvec_k = []
	for i in range(N):
		for j in range(N):
			i_k = (i*float(x)/N) + (float(x)/(2*N))
			j_k = (j*float(y)/N) + (float(y)/(2*N))
			mgd = 1e300
			for a in range(int(i_k-float(x)/(2*N)),int(i_k+float(x)/(2*N))):
				for b in range(int(j_k-float(y)/(2*N)),int(j_k+float(y)/(2*N))):
					if grad[a,b] < mgd:
						mgd  = grad[a,b]
						I, J = a, b
			cvec_k.append((img[i_k,j_k][0],
						img[i_k,j_k][1],
						img[i_k,j_k][2],i_k,j_k))
	return np.array(cvec_k)

def mag_cube(im,niveles):
	res_x = np.zeros([niveles,im.shape[0],im.shape[1]])
	res_y = np.zeros([niveles,im.shape[0],im.shape[1]])
	for k in range(niveles):
		res_x[k,:,:] = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=2*niveles+1)
		res_y[k,:,:] = cv2.Sobel(im,cv2.CV_64F,0,1,ksize=2*niveles+1)
	return res_x,res_y

def get_max_response(im_X,im_Y):
	im_mag   = np.sqrt(im_X ** 2+ im_Y ** 2)
	im_idx   = np.argmax(im_mag,axis=0)
	res_x    = np.zeros([im_X.shape[1],im_X.shape[2]])
	res_y    = np.zeros([im_X.shape[1],im_X.shape[2]])
	res_mag  = np.zeros([im_X.shape[1],im_X.shape[2]])
	for i in range(im_X.shape[1]):
		for j in range(im_X.shape[2]):
			res_x[i,j] = im_X[im_idx[i,j],i,j]
			res_y[i,j] = im_Y[im_idx[i,j],i,j]
			res_mag[i,j] = im_mag[im_idx[i,j],i,j]
	return res_x,res_y,res_mag

def SLIC(cvec,P,d,l,K,iterations):
	for it in range(iterations):
		d.fill(1e300)
		l.fill(-1)
		for k in range(K):
			x0 = cvec[k,3]
			y0 = cvec[k,4]
			lim_inf_x = max(0,x0-S)
			lim_sup_x = min(dimX-1,x0+S)
			lim_inf_y = max(0,y0-S)
			lim_sup_y = min(dimY-1,y0+S)
			for i in range(int(lim_inf_x),int(lim_sup_x)):
				for j in range(int(lim_inf_y),int(lim_sup_y)):
					dD = np.linalg.norm(P[i,j,0:3]-cvec[k,0:3]) + (m/S)*np.linalg.norm(P[i,j,3:]-cvec[k,3:])
					if dD < d[i,j] or l[i,j] == -1:
						d[i,j] = dD
						l[i,j] = k

		for k in range(K):
			x0 = cvec[k,3]
			y0 = cvec[k,4]
			lim_inf_x = max(0,x0-S)
			lim_sup_x = min(dimX-1,x0+S)
			lim_inf_y = max(0,y0-S)
			lim_sup_y = min(dimY-1,y0+S)
			mean = 0
			xvec = np.zeros(5)
			for i in range(int(lim_inf_x),int(lim_sup_x)):
				for j in range(int(lim_inf_y),int(lim_sup_y)):
					if l[i,j] == k:
						mean += 1.0
						xvec += P[i][j]
			if mean > 0:
				xvec    /= mean
				cvec[k]  = xvec

	d.fill(1e300)
	l.fill(-1)
	for k in range(K):
		x0 = cvec[k,3]
		y0 = cvec[k,4]
		lim_inf_x = max(0,x0-S)
		lim_sup_x = min(dimX-1,x0+S)
		lim_inf_y = max(0,y0-S)
		lim_sup_y = min(dimY-1,y0+S)
		for i in range(int(lim_inf_x),int(lim_sup_x)):
			for j in range(int(lim_inf_y),int(lim_sup_y)):
				dD = np.linalg.norm(P[i,j,0:3]-cvec[k,0:3]) + (m/S)*np.linalg.norm(P[i,j,3:]-cvec[k,3:])
				if dD < d[i,j] or l[i,j] == -1:
					d[i,j] = dD
					l[i,j] = k

def cls():
	os.system('cls' if os.name=='nt' else 'clear')

def slic_voronoi(factor,width,height,centers):
	image = Image.new("RGB", (factor*width, factor*height))
	putpixel = image.putpixel
	imgx, imgy = image.size
	nx = factor*centers[:,4]
	ny = factor*centers[:,3]
	nr = centers[:,0]
	ng = centers[:,1]
	nb = centers[:,2]
	for y in range(imgy):
		for x in range(imgx):
			dmin = math.hypot(imgx-1, imgy-1)
			j = -1
			for i in range(len(centers)):
				d = math.hypot(nx[i]-x, ny[i]-y)
				if d < dmin:
					dmin = d
					j = i
			putpixel((x, y), (int(nr[j]), int(ng[j]), int(nb[j])))
	image.save(path + name+"VoronoiSLIC.png", "PNG")
	plt.imshow(image)
	plt.title('Voronoi SLIC')
	plt.axis('off')
	plt.show()

#----------------------------------------------------------

# Read command line parameters:
parser = argparse.ArgumentParser(description="Demo script for pyhull.",
								 epilog="Author: Rodolfo Ferro")

parser.add_argument("-k", "--clusters", dest="k", type=int,
							default=4,
							help="Number of clusters. The square of this number will be used.")

parser.add_argument("-it", "--iterations", dest="it", type=int,
					default=1,
					help="Number of iterations for SLIC.")

parser.add_argument("-n", "--name", dest="name", type=str,
					default="frog",
					help="Name of the image.")

parser.add_argument("-f", "--factorsize", dest="factor", type=int,
					default=2,
					help="Size factor for reconstructing Voronoi.")

args = parser.parse_args()

N          = args.k
name       = args.name
iterations = args.it

#----------------------------------------------------------

# Load image:
img  = plt.imread(path + name + ".jpg")	# Read image
dimX = img.shape[0]						# Image dimensions
dimY = img.shape[1]

# Plot original image:
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Plot grad:
IMG = cv2.imread(path + name+".jpg")
b,g,r = cv2.split(IMG)
R_X, R_Y = mag_cube((r+g+b)/3.0,2)
R_X, R_Y, R_G = get_max_response(R_X,R_Y)
R_G /= 255.
plt.imshow(R_G)
plt.title('Gradient of the Image')
plt.axis('off')
plt.show()

# RGB-Matrix with (x,y) values:
print "Computing a not so stupid matrix..."
t = time.clock()
P = np.array([[np.array([1.*img[i][j][0],
	1.*img[i][j][1],1.*img[i][j][2],1.*i,1.*j]) 
	for j in range(dimY)] for i in range(dimX)])
print P.shape
print "DONE."
print "Time:", time.clock() - t, "secs.\n"

#----------------------------------------------------------

# We define the number of superpixels S:
S = max(float(dimX),float(dimY)) / N

# We set the centers vector:
cvec = cvec_k(R_G,img,dimX,dimY,N)
K    = len(cvec)

# Initialize the matrix of distances:
d = np.empty([dimX,dimY],dtype=float)
# Initialize the matrix of labels:
l = np.empty([dimX,dimY],dtype=int)

#----------------------------------------------------------

print "Computing means..."
t = time.clock()
SLIC(cvec,P,d,l,K,iterations)
print "DONE."
print "Time:", time.clock() - t, "secs.\n"

#----------------------------------------------------------

# Compute new img:
newimg = np.copy(img);
for i in range(dimX):
	for j in range(dimY):
		newimg[i,j] = [int(cvec[l[i,j],0]),int(cvec[l[i,j],1]),int(cvec[l[i,j],2])]

# Plot new image:
plt.imshow(newimg)
plt.title('SLIC with %d^2 clusters and %d iterations'%(N,iterations))
plt.axis('off')
plt.show()

#----------------------------------------------------------

# Get centers:
centers = cvec[:,3:]
plt.scatter(centers[:,1],-centers[:,0])
plt.title('Scatter of new centers')
plt.axis('off')
plt.show()

#----------------------------------------------------------

print "Computing Voronoi cells..."
t = time.clock()
slic_voronoi(args.factor,dimX,dimY,cvec)
print "DONE."
print "Time:", time.clock() - t, "secs.\n"