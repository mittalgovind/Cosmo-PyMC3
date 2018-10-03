import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import pymc3 as pm
from scipy.integrate import quad, romberg
import theano.tensor as tt
import theano
import pdb
import os

# BAO
data = open('bao1.dat', 'r')
data = list(data)
n = len(data)
zdata = np.zeros(n)
thetadata = np.zeros(n)
sigmadata = np.zeros(n)
for i in range(n):
	line = data[i].split(' ')
	zdata[i] = float(line[0])
	thetadata[i] = float(line[1])
	sigmadata[i] = float(line[2])

def integrandDa(z, H0, Om):
	h = H0/100.0
	Or = 4.16e-5/h**2
	# For LCDM, putting following line.
	fz = 1
	# fz = (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z/(1+z))
	Hz = H0*(Or*(1+z)**4 + Om*(1+z)**3 + (1 - Om - Or)*fz)**0.5
	return 1.0 / Hz
		
def integrandRs(a, H0, Om):
	h = H0/100.0
	Ob = 0.02203/h**2
	Or = 4.16e-5/h**2
	Og = 2.46e-5/h**2
	# For LCDM, putting following line.
	fa = 1
	# Use following line for arbitrary w0 and wa.
	# fa = a**(-3 * (1 + w0 + wa)) * np.exp(-3 * wa * (1 - a))
	Ha = H0 * (Or*a**-4 + Om*a**-3 + (1 - Or - Om)*fa)**0.5
	Rs = a**2 * Ha * (1 + 3*Ob*a/(4*Og))**0.5
	return 1./Rs
	  
class IntegrateOpBAO(theano.gof.Op):
	itypes=[tt.dscalar, tt.dscalar]
	otypes=[tt.dvector]
	
	def __init__(self, z, *args, **kwargs):
		super(IntegrateOpBAO, self).__init__(*args, **kwargs)
		self.z = z

	def perform(self, node, inputs, output_storage):
		H0, Om = inputs
		z = self.z
		h = H0/100.0
		Ob = 0.02203/h**2
		Or = 4.16e-5/h**2
		b1 = (0.313 * (Om * h**2)**(-0.419)) * (1 + 0.607 * (Om * h**2)**0.674)
		b2 = 0.238 * (Om * h**2)**0.223

		zd_num1 = 1291 * (Om * h**2)**0.251
		zd_num2 = 1 + b1 * (Ob * h**2)**b2
		zd_denom = 1 + 0.659 * (Om * h**2)**0.828
		zd = zd_num1 * zd_num2/zd_denom

		# For LCDM, putting following line.
		fz = 1
		# fz = (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z/(1+z))
		Hz = H0*(Or*(1+z)**4 + Om*(1+z)**3 + (1 - Om - Or)*fz)**0.5

		Da_z = np.zeros(len(z))
		for i in range(len(z)):
			Da_z[i] = quad(integrandDa, 0, z[i], args=(H0, Om))[0]
		Dv_z = (Da_z**2 * z/Hz)**(1./3)

		rs_zd = 1./(3**0.5) * quad(integrandRs, 0, 1./(1+zd), args=(H0, Om))[0]
		# pdb.set_trace()
		output_storage[0][0] = Dv_z/rs_zd


model = pm.Model()
bao = IntegrateOpBAO(zdata)

with model:
	H0 = pm.Uniform('H0', lower=55, upper=81)
	Om = pm.Uniform('Om', lower=0.1, upper=0.45)
	thetaMu = bao(H0, Om)
	obs = pm.Normal('obs', mu=thetaMu, sd=sigmadata, observed=thetadata)

	step = pm.Slice() 
	trace = pm.sample(20000, chains=2, njobs=4, step=step)
	
	print(pm.summary(trace))
	################################################################################
	### CREATING A DIRECTORY FOR CHAINS ############################################
	newpath = r'/home/hpadmin/govind/bao/bao1_chains'
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
	    fout = open('bao1_chains/chain__'
	                + str(j+1) + '.txt', "w")
	    for i in range(n):
	        fout.write(str(H0[j*n + i]) + '\t' + str(Om[j*n + i]) +'\t' + str(H0[j*n + i])
	                   + '\t' + str(Om[j*n + i]) + '\n')
	    fout.close()
	#####################################################################################
	pm.traceplot(trace)
	plt.show()
	plt.clf()
