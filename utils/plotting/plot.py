import numpy as np
import matplotlib.pyplot as plt


def watermark(t=None,logo="PANDA", px=0.033, py=0.9, fontsize=18, alpha=0.8, alpha_logo=0.95, shift=0.16, bstyle='italic', scale = 1.4, *args, **kwargs):
    """
    Args:
        t:
        logo:
        px:
        py:
        fontsize:
        alpha:
        shift:
        *args:
        **kwargs:
    Returns:
    """
    if t is None:
        import datetime
        t = " %d (Simulation)" % datetime.date.today().year
    
    scaletype = plt.gca().get_yscale()
    if scaletype == 'log':   
        bottomylim, topylim = plt.gca().get_ylim()
        plt.ylim(top=bottomylim+(topylim-bottomylim)**scale)
    else:
        bottomylim, topylim = plt.gca().get_ylim()
        plt.ylim(top=bottomylim+(topylim-bottomylim)*scale)
    
    
    plt.text(px, py, logo, ha='left',
             transform=plt.gca().transAxes,
             fontsize=fontsize,
             style=bstyle,
             alpha=alpha_logo,
             weight='bold',
             *args, **kwargs,
             # fontproperties=font,
             # bbox={'facecolor':'#377eb7', 'alpha':0.1, 'pad':10}
             )
    plt.text(px + shift, py, t, ha='left',
             transform=plt.gca().transAxes,
             fontsize=fontsize,
             #          style='italic',
             alpha=alpha,  *args, **kwargs
             #          fontproperties=font,
             # bbox={'facecolor':'#377eb7', 'alpha':0.1, 'pad':10}
             )

