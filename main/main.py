import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import pymc3 as pm
from scipy.integrate import quad
import theano.tensor as tt
import theano
import pdb
from pymc3.backends import Text
import os
import shutil

np.random.seed(432813)

# Hz
data1 = open('data/Hdata.dat', 'r')
data1 = list(data1)
n = len(data1)
zdata1 = np.zeros(n)
Hdata = np.zeros(n)
sigmadata1 = np.zeros(n)

for i in range(n):
	line = data1[i].split(' ')
	zdata1[i] = float(line[0])
	Hdata[i] = float(line[1])
	sigmadata1[i] = float(line[2])

# BAO
data2 = open('data/BAOdata.dat', 'r')
data2 = list(data2)
n = len(data2)
zdata2 = np.zeros(n)
thetadata = np.zeros(n)
sigmadata2 = np.zeros(n)
zd = 1059.6
for i in range(n):
	line = data2[i].split(' ')
	zdata2[i] = float(line[0])
	thetadata[i] = float(line[1])
	sigmadata2[i] = float(line[2])

# Supernova Ia
data = open('data/jla_mub.txt', 'r')
data = list(data)
n = len(data)
zdata = np.zeros(n)
mubdata = np.zeros(n)
for i in range(n):
	line = data[i].split(' ')
	zdata[i] = float(line[0])
	mubdata[i] = float(line[1])

data = open('data/jla_mub_covmatrix.dat', 'r')
data = list(data)
n = int(len(data)**0.5)
covmat = np.zeros((n, n))
for i in range(n):
	for j in range(n):
		covmat[i][j] = float(data[i + n*j][:-1])

# CMB
z_star = 1089.9
la_obs = 301.63
la_sd = 0.15

def integrand(z, Om):
	Om = float(Om)
	E_model = (Om * (1+z)**3 + 1 - Om)**0.5
	return 1.0 / E_model

# output mub_theo
class IntegrateOpJLA(theano.Op):
	itypes = [tt.dscalar, tt.dscalar]
	otypes = [tt.dvector]

	def __init__(self, z, *args, **kwargs):
		super(IntegrateOpJLA, self).__init__(*args, **kwargs)
		self.z = z
		self.n = len(z)

	def perform(self, node, inputs, output_storage):
		H0, Om = inputs
		z = self.z
		n = self.n
		como_dist = np.zeros(n)
		for i in range(n):
			como_dist[i] = quad(integrand, 0, z[i], args=(Om))[0]

		lumo_dist = (1 + z) * como_dist
		mub_theo = 5*np.log10(lumo_dist) - 5*np.log10(H0) + 52.38
		# pdb.set_trace()
		output_storage[0][0] = mub_theo



def integrandDa(z, H0, Om):
	H_model = H0 * (Om * (1+z)**3 + 1 - Om)**0.5
	return 1.0 / H_model
		
def integrandRs(z, H0, Om):
	Ob = 226.0/(H0*H0)
	Og = 4.15e-5
	R = 3.0*Ob/(4*Og*(1+z))
	cs =  (3.0*(1+R))**(-0.5)
	H_model = H0 * (Om * (1+z)**3 + 1 - Om)**0.5
	return cs/H_model
	  
class IntegrateOpBAO(theano.gof.Op):
	itypes=[tt.dscalar, tt.dscalar]
	otypes=[tt.dvector, tt.dvector]
	
	def __init__(self, z, zd, *args, **kwargs):
		super(IntegrateOpBAO, self).__init__(*args, **kwargs)
		self.z = z
		self.zd = zd
		
	def perform(self, node, inputs, output_storage):
		H0, Om = inputs
		z = self.z
		zd = self.zd
		rs_zd = quad(integrandRs, zd, np.inf, args=(H0, Om))[0]
		num = np.ones(len(z)) * rs_zd
		denom = np.zeros(len(z))
		for i in range(len(z)):
			denom[i] = quad(integrandDa, 0, z[i], args=(H0, Om))[0]
		output_storage[0][0] = num
		output_storage[1][0] = denom

class IntegrateOpCMB(theano.Op):
	itypes = [tt.dscalar, tt.dscalar]
	otypes = [tt.dscalar]

	def __init__(self, *args, **kwargs):
		super(IntegrateOpCMB, self).__init__(*args, **kwargs)

	def perform(self, node, inputs, output_storage):
		z_star = 1089.9
		H0, Om = inputs
		la = np.pi * quad(integrandDa, 0, z_star, args=(H0, Om))[0]/quad(integrandRs, z_star, np.inf, args=(H0, Om))[0]
		output_storage[0][0] = np.array(la)
    

class ApnaNormal(pm.Continuous):
	def __init__(self, mu, sd, *args, **kwargs):
		super(ApnaNormal, self).__init__(*args, **kwargs)
		self.mu = mu
		self.sd = sd
		self.mode = mu

	def logp(self, hvalue):
		mu = self.mu
		sd = self.sd
		logp_vector = -np.log(sd * np.sqrt(2.0 * np.pi)) - (np.sqrt(hvalue - mu) / (2.0 * sd**2))
		return logp_vector.sum() 


model = pm.Model()
bao = IntegrateOpBAO(zdata2, zd)
jla = IntegrateOpJLA(zdata)
cmb = IntegrateOpCMB()

with model:
	H0 = pm.Uniform('H0', lower=55, upper=85)
	Om = pm.Uniform('Om', lower=0.1, upper=0.5)
	
	hMu = H0 * (Om * (1+zdata1)**3 + 1 - Om)**0.5
	num, denom =bao(H0, Om)
	thetaMu = num/denom
	jlaMu = jla(H0, Om)
	laMu = cmb(H0, Om)
	#obs1 = pm.Normal('obs1', mu=hMu, sd=sigmadata1, observed=Hdata)
	obs2 = ApnaNormal('obs2', mu=thetaMu, sd=sigmadata2, observed=thetadata)
	#obs3 = pm.MvNormal('obs3', mu=jlaMu, cov=covmat, observed=mubdata, shape=(n, n))
	#obs4 = pm.Normal('obs4', mu=laMu, sd=la_sd, observed=la_obs)
	obs = obs2.sum() #+ obs2.sum() + obs3.sum() #+ obs4
	
	step = pm.Slice()
	trace = pm.sample(10000, chains=4, njobs=10, step=step)
	
	print(pm.summary(trace))
	pm.traceplot(trace)
	plt.show()
	plt.clf()

	################################################################################
	### CREATING A DIRECTORY FOR CHAINS ############################################
	newpath = r'/home/hpadmin/govind/hz_chains' 
	if not os.path.exists(newpath):
	    os.makedirs(newpath)
	#################################################################################
	#### WRITING THE CHAINS AS .txt FILES FROM THE TRACE ###########################
	H0 = trace.get_values('H0')
	Om = trace.get_values('Om')
	#Ok = trace.get_values('Ok')
	no_of_chains = len(trace.chains)
	n = int(len(H0)/no_of_chains)
	for j in range(no_of_chains):
	    fout = open('hz_chains/chain__' 
	                + str(j+1) + '.txt', "w")
	    for i in range(n):
	        fout.write(str(H0[j*n + i]) + '\t' + str(Om[j*n + i]) +'\t' + str(H0[j*n + i]) 
	                   + '\t' + str(Om[j*n + i]) + '\n')
	    fout.close()
	#####################################################################################
#print(pm.summary(trace))
#print(H_obs)
	
