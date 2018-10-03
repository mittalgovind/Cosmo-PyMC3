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
from pymc3.distributions import distribution
from pymc3.distributions.distribution import Continuous, draw_values, generate_samples
from pymc3.distributions.dist_math import bound 
from pymc3.distributions.multivariate import _QuadFormBase
from theano.tensor.nlinalg import det, matrix_inverse, trace, eigh
from theano.tensor.slinalg import Cholesky

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

data = open('sys_data.txt', 'r')
data = list(data)
cov_sys = np.zeros((n, n))
for i in range(n):
	for j in range(n):
		cov_sys[i][j] = float(data[i*n + j][:-1])

dstat = np.zeros((n, n))
for i in range(n):
	dstat[i][i] = dmb[i]**2

covmat = cov_sys + dstat

_, norm = np.linalg.slogdet(covmat)
norm *= -0.5
norm -= 100*np.log(2*np.pi) 

def integrand(z, Om):	
	Om = float(Om)
	E_model = (Om * (1+z)**3 + 1 - Om)**0.5
	return 1.0 / E_model


class DL(theano.Op):
	itypes = [tt.dscalar, tt.dscalar, tt.dscalar]
	otypes = [tt.dvector]

	def __init__(self, zcmb, zhel, *args, **kwargs):
		super(DL, self).__init__(*args, **kwargs)
		self.zcmb = zcmb
		self.zhel = zhel
		self.n = len(zcmb)

	def perform(self, node, inputs, output_storage):
		H0, Om, M = inputs
		zcmb = self.zcmb
		zhel = self.zhel
		n = self.n
		como_dist = np.zeros(n)
		for i in range(n):
			como_dist[i] = quad(integrand, 0, zcmb[i], args=(Om))[0]

		D_L = (1 + zhel) * como_dist
		m_mod = 5*np.log10(D_L) - 5*np.log10(H0) + 52.38 + M
		#m_mod = 5*np.log10(D_L) + M

		output_storage[0][0] = m_mod

model = pm.Model()
calc = DL(zcmb=zcmb_data, zhel=zhel_data)

with model:
	H0 = pm.Uniform('H0', lower=50, upper=75)
	Om = pm.Uniform('Om', lower=0.1, upper=0.5)
	M = pm.Uniform('M', lower=-21, upper=-18)

	m_mod = calc(H0, Om, M)
	# obs = pm.MvNormal('obs', mu=m_mod, cov=covmat, observed=mb_obs)
	obs = pm.Normal('obs', mu=m_mod, sd=dmb, observed=mb_data)
	
	step = pm.Metropolis()
	trace = pm.sample(25000, chains=4, cores=4, step=step, njobs=4)
	
	print(pm.summary(trace))
	pm.traceplot(trace)
	plt.show()
	plt.clf()

################################################################################
### CREATING A DIRECTORY FOR CHAINS ############################################
newpath = r'/home/hpadmin/govind/pantheon/skpan_chains' 
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
    fout = open('skpan_chains/chain__' 
                + str(j+1) + '.txt', "w")
    for i in range(n):
        fout.write(str(H0[j*n + i]) + '\t' + str(Om[j*n + i]) +'\t' + str(H0[j*n + i]) 
                   + '\t' + str(Om[j*n + i]) + '\n')
    fout.close()
#####################################################################################
#print(pm.summary(trace))
#print(H_obs)
