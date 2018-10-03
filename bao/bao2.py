import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import pymc3 as pm
from scipy.integrate import quad, romberg
import theano.tensor as tt
import theano
import pdb
from pymc3.backends import Text
import os
import shutil

z_data = np.array([0.57, 0.35])
H_data = np.array([87.6, 81.3])
DA_data = np.array([1396, 1037])
Omh2_data = np.array([0.126, 0.1268])

cov = np.array([
		[	
			[0.0385, -0.001141, -13.53],
			[-0.001141, 0.0008662, 3.354],
			[-13.53, 3.354, 19370]
		],
		[
			[0.00007225, -0.169609, 0.01594328],
			[-0.169609, 1936, 67.03048],
			[0.01594328, 67.03048, 14.44]
		]
	])

def integrandDa(z, H0, Om):
	h = H0/100.0
	Or = 4.16e-5/h**2
	Hz = H0*(Or*(1+z)**4 + Om*(1+z)**3 + (1 - Om - Or))**0.5
	return 1.0 / Hz


class IntegrateOpDa(theano.gof.Op):
	itypes=[tt.dscalar, tt.dscalar]
	otypes=[tt.dvector]
	
	def __init__(self, z, *args, **kwargs):
		super(IntegrateOpDa, self).__init__(*args, **kwargs)
		self.z = z

	def perform(self, node, inputs, output_storage):
		H0, Om = inputs
		z = self.z
		Da = np.zeros(len(z))
		c = 299792458.0/1000
		Da[0] = c * quad(integrandDa, 0, z[0], args=(H0, Om))[0]/(1+z[0])
		Da[1] = c * quad(integrandDa, 0, z[1], args=(H0, Om))[0]/(1+z[1])
		output_storage[0][0] = Da

model = pm.Model()
DA = IntegrateOpDa(z_data)

with model:
	H0 = pm.Uniform('H0', lower=55, upper=80)
	Om = pm.Uniform('Om', lower=0.1, upper=0.5)

	h = H0/100.0
	Or = 4.16e-5/h**2
	H_model = np.zeros(len(z_data))
	H_model = H0*[(Or*(1+z_data[0])**4 + Om*(1+z_data[0])**3 + (1 - Om - Or))**0.5,
					(Or*(1+z_data[1])**4 + Om*(1+z_data[1])**3 + (1 - Om - Or))**0.5]

	DA_model = DA(H0, Om)

	h = H0/100.0
	Omh2_model = Om * h**2

	mu_model = [[H_model[0], DA_model[0], Omh2_model], 
				[H_model[1], DA_model[1], Omh2_model]]

	obs_data = [[H_data[0], DA_data[0], Omh2_data[0]], 
				[H_data[1], DA_data[1], Omh2_data[1]]]

	obs = [pm.MvNormal('obs0', mu=mu_model[0], cov=cov[0], observed=obs_data[0]),
			pm.MvNormal('obs1', mu=mu_model[1], cov=cov[1], observed=obs_data[1])]

	main_likelihood = sum(obs)
	folder_name = 'bao_chains'
	try:
		shutil.rmtree('./' + folder_name)
	except:
		pass
	backend = Text(folder_name)

	step = pm.Slice() 
	trace = pm.sample(20000, chains=2, trace=backend, step=step)
	
	newpath = r'/home/hpadmin/govind/bao/bao2_chains'
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
	    fout = open('bao2_chains/chain__'
	                + str(j+1) + '.txt', "w")
	    for i in range(n):
	        fout.write(str(H0[j*n + i]) + '\t' + str(Om[j*n + i]) +'\t' + str(H0[j*n + i])
	                   + '\t' + str(Om[j*n + i]) + '\n')
	    fout.close()
	#####################################################################################

	pm.traceplot(trace)
	plt.show()
	plt.clf()
