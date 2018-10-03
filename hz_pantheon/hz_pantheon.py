import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import pymc3 as pm
from scipy.integrate import quad
import theano.tensor as tt
import theano
import pdb
import os

# -------------- Hz data input ----------
data = open('Hdata.dat', 'r')
data = list(data)
n = len(data)
z_data = np.zeros(n)
H_data = np.zeros(n)
Hcov = np.zeros((n, n))

for i in range(n):
    line = data[i].split(' ')
    z_data[i] = float(line[0])
    H_data[i] = float(line[1])
    Hcov[i][i] = float(line[2][:-1])**2

# put some non-zero points like this
Hcov[0][1] = 1.78
Hcov[0][2] = 0.93
Hcov[1][0] = 1.78
Hcov[1][2] = 2.20
Hcov[2][0] = 0.93
Hcov[2][1] = 2.20

# ---------- pantheon data input --------
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

_, norm = np.linalg.slogdet(covmat)
norm *= -0.5
norm -= 100*np.log(2*np.pi) 

def integrand(z, Om):	
	Om = float(Om)
	E_model = (Om * (1+z)**3 + 1 - Om)**0.5
	return 1.0 / E_model


class DL(theano.Op):
	itypes = [tt.dscalar, tt.dscalar]
	otypes = [tt.dvector]

	def __init__(self, zcmb, zhel, *args, **kwargs):
		super(DL, self).__init__(*args, **kwargs)
		self.zcmb = zcmb
		self.zhel = zhel
		self.n = len(zcmb)

	def perform(self, node, inputs, output_storage):
		Om, M = inputs
		zcmb = self.zcmb
		zhel = self.zhel
		n = self.n
		como_dist = np.zeros(n)
		for i in range(n):
			como_dist[i] = quad(integrand, 0, zcmb[i], args=(Om))[0]

		D_L = (1 + zhel) * como_dist
		m_mod = 5*np.log10(D_L) + 52.38 + M

		output_storage[0][0] = m_mod

model = pm.Model()
calc = DL(zcmb=zcmb_data, zhel=zhel_data)

with model:
	H0 = pm.Uniform('H0', lower=55, upper=81)
	Om = pm.Uniform('Om', lower=0.1, upper=0.5)
	M = pm.Uniform('M', lower=-40, upper=0)

	m_mod = calc(Om, M)
	pantheon_likelihood = pm.MvNormal('obs', mu=m_mod, cov=covmat, observed=mb_data)

	H_model = H0 * (Om * (1+z_data)**3 + 1 - Om)**0.5
	Hz_likelihood = pm.MvNormal('Obs', mu=H_model, cov=Hcov, observed=H_data)
	
	obs = pantheon_likelihood.sum() + Hz_likelihood.sum()

	step = pm.Metropolis()
	trace = pm.sample(40000, chains=4, step=step, njobs=4)
	
	print(pm.summary(trace))
	################################################################################
	### CREATING A DIRECTORY FOR CHAINS ############################################
	newpath = r'/home/hpadmin/govind/hz_pantheon/hz_pantheon_chains' 
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
	    fout = open('hz_pantheon_chains/chain__' 
	                + str(j+1) + '.txt', "w")
	    for i in range(n):
	        fout.write(str(H0[j*n + i]) + '\t' + str(Om[j*n + i]) +'\t' + str(H0[j*n + i]) 
	                   + '\t' + str(Om[j*n + i]) + '\n')
	    fout.close()
	#####################################################################################

	pm.traceplot(trace)
	plt.show()
	plt.clf()
