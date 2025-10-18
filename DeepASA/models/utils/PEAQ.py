## PEAQ.py
##  
## PEAQ.py is our overall class implementation of the PEAQ algorihtm. We are leaving big parts 
## out, and taking a significanly different appraoch to overall architechture than Kabal 2002.
##
## Matthew Cohen and Stephen Welch
## Version 1.0

import numpy as np
from models.utils.PQEval import PQEval
import time

class PEAQ(object):
	def __init__(self, Amax = 1, Fs = 16000, NF = 2048):
		# Amax = maximum signal amplitude
		# Fs = sampling frequency
		# NF = Length of analysis window

		self.NF = NF
		self.Fs = Fs
		self.Amax = Amax

		#Step forward in half window lengths:
		self.Nadv = self.NF / 2

		#Number of critical bands:
		self.Nc = 109

	def process(self, testSignal, referenceSignal):
		#Preform basic procssing (Section 2 in Kabal.)
		# sigR = reference signal	
		# sigT = test signal

		sigR = referenceSignal
		sigT = testSignal

		#Number of frames:
		self.Np = int(np.floor(len(sigR)/self.Nadv)-1)
		
		#Scale audio:
		if np.amax(abs(sigR)) != self.Amax:
			sigRS = self.Amax*sigR/float(np.amax(abs(sigR)))
			sigTS = self.Amax*sigT/float(np.amax(abs(sigT)))
			# print('Signals scaled, max reference value = ' + str(np.amax(abs(sigRS))) + ',')
			# print('and max test value = ' + str(np.amax(abs(sigTS))) +'.')

		#Instantiate Object to process single frames of data:
		self.PQE = PQEval(Amax = self.Amax, Fs = self.Fs, NF = self.NF)

		# print('Processing Audio...')
		
		#Create empty matrices:
		X2 = np.zeros((2,self.NF//2+1))

		self.X2MatR = np.zeros((self.Np, self.NF//2+1))
		self.X2MatT = np.zeros((self.Np, self.NF//2+1))

		self.EbNMat = np.zeros((self.Np, self.Nc))
		self.EsMatR = np.zeros((self.Np, self.Nc))
		self.EsMatT = np.zeros((self.Np, self.Nc))

		self.EhsR = np.zeros((self.Np, self.Nc))
		self.EhsT = np.zeros((self.Np, self.Nc))

		previousFrameR = np.zeros(self.Nc)
		previousFrameT = np.zeros(self.Nc)

		#Maybe take this out later, but useful in debugging:
		self.xMatR = np.zeros((self.Np, self.NF))
		self.xMatT = np.zeros((self.Np, self.NF))

		startS = 0

		for i in np.arange(self.Np):
			xR = sigRS[startS:self.NF+startS]
			xT = sigTS[startS:self.NF+startS]
			
			startS = int(startS+self.Nadv)
			self.xMatR[i, :] = xR
			self.xMatT[i, :] = xT
			
			X2[0,:] = self.PQE.PQDFTFrame(xR)
			X2[1,:] = self.PQE.PQDFTFrame(xT)
			
			self.X2MatR[i,:] = X2[0,:]
			self.X2MatT[i,:] = X2[1,:]
			
			self.EbN, self.Es = self.PQE.PQ_excitCB(X2)
			self.EbNMat[i,:] = self.EbN
			self.EsMatR[i,:] = self.Es[0,:]
			self.EsMatT[i,:] = self.Es[1,:]
			
			self.EhsR[i,:], previousFrameR = self.PQE.PQ_timeSpread(self.EsMatR[i,:], previousFrameR)
			self.EhsT[i,:], previousFrameT = self.PQE.PQ_timeSpread(self.EsMatT[i,:], previousFrameT)
			self.xMatR[i, :] = xR
			self.xMatT[i, :] = xT
			X2[0,:] = self.PQE.PQDFTFrame(xR)
			X2[1,:] = self.PQE.PQDFTFrame(xT)
			self.X2MatR[i,:] = X2[0,:]
			self.X2MatT[i,:] = X2[1,:]
			self.EbN, self.Es = self.PQE.PQ_excitCB(X2)
			self.EbNMat[i,:] = self.EbN
			self.EsMatR[i,:] = self.Es[0,:]
			self.EsMatT[i,:] = self.Es[1,:]
			self.EhsR[i,:], previousFrameR = self.PQE.PQ_timeSpread(self.EsMatR[i,:], previousFrameR)
			self.EhsT[i,:], previousFrameT = self.PQE.PQ_timeSpread(self.EsMatT[i,:], previousFrameT)

		# print('Audio Processed! (Kabal, section 2), ' + str(self.Np) + ' windows processed ')
		BWRef, BWTest = self.computeBW(self.X2MatR, self.X2MatT)
		NMRavg, NMRmax = self.computeNMR(self.EbNMat, self.EhsR)
		
		totalNMRB, relDistFramesB = self.PQ_avgNMRB(NMRavg, NMRmax)
		BandwidthRefB, BandwidthTestB = self.PQ_avgBW(BWRef, BWTest)

        # ODG 점수 계산
		ODG = self.calculate_ODG(totalNMRB, BandwidthRefB, BandwidthTestB)
		return ODG
	
	def calculate_ODG(self, NMRB, BWRefB, BWTestB):
		print('NMRB:', NMRB)
		print('BWRefB:', BWRefB)
		print('BWTestB:', BWTestB)

		ODG = sigmoid(BWRefB - BWTestB)
		ODG = np.clip(np.nan_to_num(ODG), -4, 0)  # ODG 범위는 -4에서 0 사이입니다.
		print('ODG:', ODG)
		return ODG

	def computeBW(self, X2MatR, X2MatT):
		#Kabal Section 5.4
		BWRef = np.zeros(self.Np)
		BWTest = np.zeros(self.Np)

		for i in range(int(self.Np)):
			BWRef[i], BWTest[i] = self.PQmovBW(np.vstack((self.X2MatR[i,:], self.X2MatT[i,:])))

		return BWRef, BWTest

	def PQmovBW(self, X2):
		fx = 21586
		# kx = int(round(self.NF * float(fx)/self.Fs)) # 921
		kx = 921
		fl = 8109
		# kl = int(round(self.NF * float(fl)/self.Fs)) # 346
		kl = 346
		FRdB = 10 # Ref. signal to exceed threshold level by 10dB
		FR = 10**(FRdB/10.) #added dot to make floating point - SW
		FTdB = 5 # Test signal to exceed threshold level by 5dB
		FT = 10**(FTdB/10.) #added dot to make floating point - SW
		N = self.NF//2
		Xth = np.amax(X2[1,kx:N])
		BWRef = -1
		XthR = FR * Xth # Reference signal threshold level
		for k in range(kx-1,kl,-1):
			if (X2[0,k+1] >= XthR):
				BWRef = k+1
				break
		
		BWTest = -1
		XthT = FT * Xth # Test signal threshold level
		for k in range(BWRef-1,-1,-1):
			if (X2[1,k+1] >= XthT):
				BWTest = k+1
				break
		return BWRef, BWTest

	def computeNMR(self, EbNMat, EhsR):
		#Kabal Section ...
		#Compute NRM for whole time series.

		NMRavg = np.zeros(self.Np)
		NMRmax = np.zeros(self.Np)

		for i in range(int(self.Np)):
			NMR = self.PQmovNMRB(EbNMat[i,:], EhsR[i,:])
			NMRavg[i] = NMR['NMRavg']
			NMRmax[i] = NMR['NMRmax']

		return NMRavg, NMRmax

	def PQmovNMRB(self, EbN, Ehs):
		NMR = dict()
		Nc, fc, fl, fu, dz = self.PQE.PQCB()
		gm = self.PQ_MaskOffset(dz, Nc)
		
		NMRmax = 0
		NMRm = 0
		s = 0
		
		R_NM = np.zeros(Nc)
		for k in range(Nc):
			NMRm = EbN[k] / (gm[k] * Ehs[k])
			R_NM[k] = NMRm # Remove later!
			s = s + NMRm
			
			if (NMRm > NMRmax):
				NMRmax = NMRm
				
		NMR['NMRmax'] = NMRmax
		NMR['NMRavg'] = float(s)/Nc
		
		return NMR

	def PQ_MaskOffset(self, dz, Nc):
		gm = np.zeros(Nc)
		for k in range(Nc):
			if (k <= 12./dz):
				mdB = 3
			else:
				mdB = 0.25*k*dz
			gm[k] = 10**(-1*float(mdB)/10)
		return gm


	## --------------- Averaging -------------------- ##
	## Time averaging functions for MOVs
	## Same naming convention as Kabal
	##

	def PQ_avgBW(self, BWRef, BWTest):
		# I think this is just an average of all the 
		# positive values, as far as I can tell...
		# Our implementation is simpler too, becuase we aren't worried about stereo
		# Ok, so these values don't exactly match Octave, but they are pretty close (+)
		BandwidthRefB = np.mean(BWRef[BWRef >=0])
		BandwidthTestB = np.mean(BWTest[BWTest >=0])

		return BandwidthRefB, BandwidthTestB

	def PQ_avgNMRB(self, NMRavg, NMRmax):
		#Average NMR values, we also get another MOV here for free - RelDistFramesB
		#This has been validated against Octave, appears to match very well.

		totalNMRB = 10*np.log10(np.mean(NMRavg))

		#Threshold:
		Tr = 10**(1.5/10)
		relDistFramesB = np.mean(NMRmax>Tr)

		return totalNMRB, relDistFramesB