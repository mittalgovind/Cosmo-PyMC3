from __future__ import print_function
from getdist import plots, MCSamples
from getdist import loadMCSamples
import warnings

warnings.filterwarnings('ignore', '.*tight_layout.*',)
import numpy as np

###### CREATING .paramnames FILE FOR GETDIST
with open('/home/hpadmin/govind/pantheon/skpan_chains/chain_.paramnames','w') as f:
    f.write("H0 H_0 \n Omega_m \Omega_{\\rm m}")
##########################################################################################################


samples1 = loadMCSamples('/home/hpadmin/govind/pantheon/skpan_chains/chain_', 
                         dist_settings={'ignore_rows':0.3})

g = plots.getSubplotPlotter(width_inch=7.0)
g.settings.axes_fontsize = 12.0
g.settings.lab_fontsize = 12.0
g.settings.figure_legend_frame = True
samples1.contours = np.array([0.68, 0.95, 0.99])

samples1.updateBaseStatistics()
g.settings.num_plot_contours = 2

g.triangle_plot([samples1],['Omega_m','H0'],filled=True,contour_colors=['b'])
#g.add_legend(['Hubble data only'],legend_loc = 'best')

#g.export('/home/santosh/Desktop/triangle.pdf')
g.export('triangle.pdf')
print(samples1.getTable(limit=1).tableTex())

