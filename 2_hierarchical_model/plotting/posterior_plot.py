import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
from matplotlib.figure import Figure

from .HPD_regions import HPD_contours

class PosteriorPlot(object):
    """
    Simple 2D posterior plots.
    """

    def __init__(self, chain1, chain2, prior1=None, prior2=None, levels=[0.99, 0.9, 0.6, 0.3],
                 color=cm.viridis(0.0), alpha=1.0, xlim=None, ylim=None, fig=None, ax=None,
                 marginals=True, vlim=0.25, bins=60, clip=None):
        """
        Simple 2D posterior plots.

        :param chain1: MCMC chain of parameter1 values
        :param chain2: MCMC chain of parameter2 values
        :param prior1: Prior samples for parameter1
        :param prior2: Prior samples for parameter2
        :param levels: Desired levels for HPD regions
        """

        self.chain1 = chain1

        self.chain2 = chain2

        self.prior1 = prior1

        self.prior2 = prior2

        self.levels = levels

        self.color = color

        self.alpha = alpha

        self.xlim = xlim

        self.ylim = ylim

        self.fig = fig

        self.ax = ax

        self.vlim = vlim   

        self.bins = bins
        
        self.marginals = marginals

        self.clip = clip
        
        self._calculate_levels()

        self._plot()

        
        
    def _calculate_levels(self):
    
        self.hpd_levels = HPD_contours(self.chain1, self.chain2, levels=self.levels, bins=self.bins, kde=True)
        

    def _plot(self):

        if isinstance(self.fig, Figure):

            pass

        else:
        
            gs_kw = dict(width_ratios=[3, 1], height_ratios=[1, 3])
        
            self.fig, self.ax = plt.subplots(ncols=2, nrows=2, constrained_layout=False, gridspec_kw=gs_kw)

            self.fig.set_size_inches((6, 5))

            self.ax[1,0].spines['top'].set_visible(True)
            self.ax[1,0].spines['right'].set_visible(True)
            self.ax[0,0].spines['right'].set_visible(True)
            self.ax[0,0].spines['top'].set_visible(True)
            self.ax[0,0].get_yaxis().set_visible(False)
            self.ax[0,0].get_xaxis().set_ticklabels([])
            self.ax[1,1].spines['top'].set_visible(True)
            self.ax[1,1].spines['right'].set_visible(True)
            self.ax[1,1].get_xaxis().set_visible(False)
            self.ax[1,1].get_yaxis().set_ticklabels([])    
            self.ax[0,1].set_axis_off()
            if not self.marginals:
                self.ax[0,0].set_axis_off()
                self.ax[1,1].set_axis_off()
            self.fig.tight_layout()
            self.fig.subplots_adjust(hspace=0.035, wspace=0.03)

       
        # Main 2D KDE plot
        sns.kdeplot(self.chain1, self.chain2, shade=True, shade_lowest=True, ax=self.ax[1,0],
                    n_levels=self.hpd_levels, color=self.color, alpha=self.alpha, clip=self.clip)
        self.ax[1,0].set_ylim(self.ylim)
        self.ax[1,0].set_xlim(self.xlim)

        if self.marginals:
            Nbins_x = 30
            Nbins_y = 20
            # Chain 1 marginal plot
            #sns.kdeplot(self.chain1, shade=True, ax=self.ax[0,0], color=self.color)
            self.ax[0,0].hist(self.chain1, histtype='step', color=self.color, density=True, lw=2,
                              bins=np.linspace(self.xlim[0], self.xlim[1], Nbins_x))
            self.ax[0,0].hist(self.chain1, color=self.color, density=True, alpha=0.2, lw=0,
                              bins=np.linspace(self.xlim[0], self.xlim[1], Nbins_x))
            if isinstance(self.prior1, np.ndarray):
                sns.kdeplot(self.prior1, shade=False, ax=self.ax[0,0], color='k',linestyle=':')
            self.ax[0,0].set_xlim(self.xlim)
            self.ax[0,0].set_ylim(0, self.vlim)
        
            # Chain 2 marginal plot
            #sns.kdeplot(self.chain2, shade=True, ax=self.ax[1,1], color=self.color, vertical=True)
            self.ax[1,1].hist(self.chain2, histtype='step', color=self.color, density=True, lw=2,
                              bins=np.linspace(self.ylim[0], self.ylim[1], Nbins_y), orientation='horizontal')
            self.ax[1,1].hist(self.chain2, color=self.color, density=True, alpha=0.2, lw=0,
                              bins=np.linspace(self.ylim[0], self.ylim[1], Nbins_y), orientation='horizontal')
            if isinstance(self.prior2, np.ndarray):
                sns.kdeplot(self.prior2, shade=False, ax=self.ax[1,1], color='k', linestyle=':', vertical=True)
            self.ax[1,1].set_ylim(self.ylim)
            self.ax[1,1].set_xlim(0, self.vlim)
        
    def save(self, filename):

        self.fig.savefig(filename, dpi=500, bbox_inches='tight')
            
        
        
