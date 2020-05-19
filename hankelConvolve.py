
#import relevant modules
import numpy as np
from hankel import HankelTransform
from scipy.interpolate import InterpolatedUnivariateSpline as spline

class hankelConvolve():
	k = np.logspace(-6,1,50) #k range for transform
	hankel1 = HankelTransform(nu=0, N=200, h=0.01) #hankel transform for 1st function
	hankel2 = HankelTransform(nu=0, N=200, h=0.01) #hankel transform for 2nd function
	hankelInv = HankelTransform(nu=0, N=200, h=0.01) #hankel transform inverse transformation

	def __init__(self, f1, f2, k=None, N=None, h=None):
		self.f1 = f1
		self.f2 = f2
		if k is not None: self.k = k

		if N is not None and h is not None: #if there is both N input and h input
			if self.isiter(N): #check if N is iterable
				try: #try to unpack values
					N1,N2,N3=N
				except ValueError: #if incorrect number of values
					print('Less than 3 values in N array')
					raise
			else: #if N is single value
				N1=N2=N3=N
			if self.isiter(h): #check if h is iterable
				try: #try to unpack values
					h1,h2,h3=h
				except ValueError: #if incorrect number of values
					print('Less than 3 values in h array')
					raise
			else: #if h is single value
				h1=h2=h3=h

			#create hankel transform objects
			self.hankel1 = HankelTransform(nu=0, N=N1, h=h1)
			self.hankel2 = HankelTransform(nu=0, N=N2, h=h2)
			self.hankelInv = HankelTransform(nu=0, N=N3, h=h3)

	def isiter(self, x): #check if object is iterable
		try:
			iterable = iter(x)
			return True
		except TypeError:
			return False

	def convolve(self, r): #Hankel convolve two functions
		hf1 = 2*np.pi*self.hankel1.transform(self.f1, self.k, ret_err=False)
		hf2 = 2*np.pi*self.hankel2.transform(self.f2, self.k, ret_err=False)

		hs = spline(self.k, hf1*hf2)
		f_out = self.hankelInv.transform(hs, r, False, inverse=True)/(2*np.pi)
		return f_out


