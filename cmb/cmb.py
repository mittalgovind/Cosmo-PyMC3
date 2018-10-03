import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import pymc3 as pm
from scipy.integrate import quad
import theano.tensor as tt
import theano
import pdb
import os
import shutil

la_obs = 301.63
la_sd = 0.15

def integrandR(z, H0, Om):
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

class IntegrateOpCMB(theano.Op):
	itypes = [tt.dscalar, tt.dscalar]
	otypes = [tt.dscalar]

	def __init__(self, *args, **kwargs):
		super(IntegrateOpCMB, self).__init__(*args, **kwargs)

	def perform(self, node, inputs, output_storage):
		z_star = 1089.9
		H0, Om = inputs
		num = quad(integrandR, 0, z_star, args=(H0, Om))[0]
		denom = 1./(3**0.5) * quad(integrandRs, 0, 1./(1+z_star), args=(H0, Om))[0]
		la = np.pi * num/denom
		# pdb.set_trace()
		output_storage[0][0] = np.array(la)

cmb = IntegrateOpCMB()
model = pm.Model()

with model:
	H0 = pm.Uniform('H0', lower=55, upper=82)
	Om = pm.Uniform('Om', lower=0.1, upper=0.5)

	laMu = cmb(H0, Om)

	obs = pm.Normal('obs', mu=laMu, sd=la_sd, observed=la_obs)
	step = pm.Slice()
	trace = pm.sample(10000, chains=2, step=step)
	print(pm.summary(trace))

	################################################################################
	### CREATING A DIRECTORY FOR CHAINS ############################################
	newpath = r'/home/hpadmin/govind/cmb/cmb_chains' 
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
	    fout = open('cmb_chains/chain__' 
	                + str(j+1) + '.txt', "w")
	    for i in range(n):
	        fout.write(str(H0[j*n + i]) + '\t' + str(Om[j*n + i]) +'\t' + str(H0[j*n + i]) 
	                   + '\t' + str(Om[j*n + i]) + '\n')
	    fout.close()
	#####################################################################################

	pm.traceplot(trace)
	plt.show()
	plt.clf()
