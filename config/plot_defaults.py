from config.imports import * 

"""
==================================================================================================================
Plotting Defaults
==================================================================================================================
"""

# select plot stylesheet
plt.style.use("bmh")

font = {
    #'family' : 'normal',
    #'weight' : 'bold',
    'size' : 12
    }
matplotlib.rc('font', **font)

figure_settings = {'figsize' : (10,6)}

plot_settings = {
    'markersize' : 2
    }

legend_settings = {}

# full width image in paper
tikz_settings = {
    'axis_width' : '160mm',
    'standalone' : True
    }