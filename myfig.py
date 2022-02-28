from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pylab as plt
plt.rcParams['axes.linewidth'] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
Fontsize = 12; SmallFont = 12; SmallerFont = 10; TickPad = 3
MajorTickSize = 8; MajorTickWidth = 2.5; MinorTickSize = 5
MinorTickWidth = 25; MinorTickInterval = 2; AxisLabelWidth = 2.5
XLabelPad = 8; YLabelPad = 8
font = {'fontname':'Helvetica','color':'black','fontweight':'bold',
        'fontsize':Fontsize}
plt.rcParams['mathtext.default']='regular'
plt.rc(("xtick.major"), pad=TickPad); plt.rc(("ytick.major"), pad=TickPad)
plt.rc("axes", linewidth=AxisLabelWidth)
plt.rc("lines", markeredgewidth=MajorTickWidth)

ax = plt.subplot(111)    
ax.tick_params(which='major',width=MajorTickWidth,length=MajorTickSize,labelsize=SmallFont)
ax.tick_params(which='minor',width=MinorTickWidth,length=MinorTickSize)
ax.tick_params(which='both',direction="in",top=True,right=True)
plt.xlabel("...", labelpad=XLabelPad, fontsize=Fontsize)
plt.ylabel("...", labelpad=YLabelPad, fontsize=Fontsize)
