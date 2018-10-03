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

data = open('pantheon.dat', 'r')
data = list(data)
n = len(data)
zcmb_data = np.zeros(n)
zhel_data = np.zeros(n)
mb_data = np.zeros(n)
dmb = np.zeros(n)

for i in range(n):
	line = data[i].split(' ')
	zcmb_data[i] = float(line[1])
	zhel_data[i] = float(line[2])
	mb_data[i] = float(line[4])
	dmb[i] = float(line[5])

data = open('sys_data.dat', 'r')
data = list(data)
cov_sys = np.zeros((n, n))
for i in range(n):
	for j in range(n):
		cov_sys[i][j] = float(data[i*n + j][:-1])

dstat = np.zeros((n, n))
for i in range(n):
	dstat[i][i] = dmb[i]**2

covmat = cov_sys + dstat

def integrand(z, H0, Om):
	h = H0/100.0
	Or = 4.16e-5/h**2
	# For LCDM, putting following line.
	fz = 1
	# fz = (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z/(1+z))
	Hz = (Or*(1+z)**4 + Om*(1+z)**3 + (1 - Om - Or)*fz)**0.5
	return 1.0 / Hz

class IntegrateOpPantheon(theano.Op):
	itypes = [tt.dscalar, tt.dscalar, tt.dscalar]
	otypes = [tt.dvector]

	def __init__(self, zcmb, zhel, *args, **kwargs):
		super(IntegrateOpPantheon, self).__init__(*args, **kwargs)
		self.zcmb = zcmb
		self.zhel = zhel
		self.n = len(zcmb)

	def perform(self, node, inputs, output_storage):
		H0, Om, M = inputs
		zcmb = self.zcmb
		zhel = self.zhel
		n = self.n
		como_dist = np.ones(n)
		for i in range(n):
			como_dist[i] = quad(integrand, 0, zcmb[i], args=(H0, Om))[0]

		D_L = (1 + zhel) * como_dist
		m_mod = 5*np.log10(D_L) + M

		output_storage[0][0] = m_mod

model = pm.Model()
jla = IntegrateOpPantheon(zcmb=zcmb_data, zhel=zhel_data)

with model:
	H0 = pm.Uniform('H0', lower=55, upper=82)
	Om = pm.Uniform('Om', lower=0.1, upper=0.5)
	M = pm.Uniform('M', lower=10, upper=50)

	m_mod = jla(H0, Om, M)
	obs = pm.MvNormal('obs', mu=m_mod, cov=covmat, observed=mb_data)
	
	step = pm.Slice()
	trace = pm.sample(40000, chains=4, step=step, njobs=4)

	################################################################################
	### CREATING A DIRECTORY FOR CHAINS ############################################
	newpath = r'/home/hpadmin/govind/pantheon/pantheon_chains' 
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
	    fout = open('pantheon_chains/chain__' 
	                + str(j+1) + '.txt', "w")
	    for i in range(n):
	        fout.write(str(H0[j*n + i]) + '\t' + str(Om[j*n + i]) +'\t' + str(H0[j*n + i]) 
	                   + '\t' + str(Om[j*n + i]) + '\n')
	    fout.close()
	#####################################################################################
	print(pm.summary(trace))
	pm.traceplot(trace)
	plt.show()
	plt.clf()
