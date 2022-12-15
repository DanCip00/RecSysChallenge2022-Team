import os 
import sys
while os.path.split(os.getcwd())[1] != 'RecSysChallenge2023-Team':
    os.chdir('..')
sys.path.insert(1, os.getcwd())

import Daniele.Utils.MyDataManager as dm
import Daniele.Utils.MatrixManipulation as mm


mm.defaultExplicitURM(dm.getURMviews(),dm.getURMopen(),icml=dm.getICMl(),icmt=dm.getICMt(),appendICM=True)