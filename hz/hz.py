import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import pymc3 as pm
from scipy.integrate import quad
import theano.tensor as tt
import theano
import pdb
import os
np.random.seed(432813)

data = open('Hdata.dat', 'r')
data = list(data)
n = len(data)
z_data = np.zeros(n)
H_data = np.zeros(n)
cov = np.zeros((n, n))

for i in range(n):
    line = data[i].split(' ')
    z_data[i] = float(line[0])
    H_data[i] = float(line[1])
    cov[i][i] = float(line[2][:-1])**2

# put some non-zero points like this
cov[0][1] = 1.78
cov[0][2] = 0.93
cov[1][0] = 1.78
cov[1][2] = 2.20
cov[2][0] = 0.93
cov[2][1] = 2.20

model = pm.Model()

with model:
	H0 = pm.Uniform('H0', lower=50, upper=100)
	Om = pm.Uniform('Om', lower=0.0, upper=0.5)
	H_model = H0 * (Om * (1+z_data)**3 + 1 - Om)**0.5
	obs = pm.MvNormal('Obs', mu=H_model, cov=cov, observed=H_data)

	folder_name = 'hz_chains'
	try:
		shutil.rmtree('./' + folder_name)
	except:
		pass
	backend = Text(folder_name)

	step = pm.NUTS() 
	trace = pm.sample(20000, chains=4, trace=backend, step=step)
	
	print(pm.summary(trace))
	i = 1
	os.chdir('./' + folder_name) 
	for filename in os.listdir('.'):
		f = open(filename, 'r')
		f = list(f)
		f = f[1:]
		out = open('chain__' + str(i) + '.txt', 'w')
		for line in f:
			line = line.replace(',', '\t')
			out.write(line)
		out.close()
		os.remove(filename)
		i += 1

	os.chdir('./..')

	pm.traceplot(trace)
	plt.show()
	plt.clf()

	################################################################################
	### CREATING A DIRECTORY FOR CHAINS ############################################
	newpath = r'/home/hpadmin/govind/hz/hz_chains'
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
