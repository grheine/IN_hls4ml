import uproot
import pandas as pd

def load_root(path, start=0, end=None, showentries=True):
    print('Reading Root files, this may take a while')
    #Load simulation and digitization data from root files
    path += "_{}.root"
    file_sim = uproot.open(path.format("sim")) #MC event generator + transport + detector simulation
    file_digi = uproot.open(path.format("digi")) #digitization (like real detector answer)
    ftspoints = file_sim["pndsim"]["FTSPoint"] 
    ftshits = file_digi["pndsim"]["FTSHit"] 

    #show all entries
    if showentries:
        ftspoints.show()
        ftshits.show()
    
    #Load needed features (from detector answer and MC truth) in pandas dataframe
    digifeatures = ["X", "Z", "Isochrone", "LayerID", "ChamberID", "Skewed", "TubeID"]
    simfeatures = ["X", "Y", "Z", "TrackID", "Px", "Py", "Pz"]
    digifeatures[:] = ["FTSHit.f" + x for x in digifeatures]
    simfeatures[:] = ["FTSPoint.f" + x for x in simfeatures]

    hits = ftshits.arrays(digifeatures, library="pd", entry_start = start)
    points = ftspoints.arrays(simfeatures, library="pd", entry_stop = end)
    df = pd.concat([hits, points], axis=1)

    #rename labels and return
    df.index.names = ['event_id', 'hit_id']
    df.columns = ['x', 'z', 'iso', 'layer_id', 'chamber_id', 'skewed', 'tube_id', 'tx', 'ty', 'tz', 'particle_id', 'px', 'py', 'pz']
    df = df.reset_index('hit_id')
    df = df[['hit_id', 'x', 'z', 'iso', 'pz', 'chamber_id', 'skewed', 'particle_id', 'layer_id']]
    return df.loc[start:end]