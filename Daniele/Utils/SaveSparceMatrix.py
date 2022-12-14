import scipy.sparse as sps
import pandas as pd

def saveMatrix (dir,csr):

    coo = csr.tocoo()
    frame = { coo.shape[0] : coo.row, coo.shape[1] : coo.col, 'data': coo.data}
    pd.DataFrame(frame).to_csv(dir, index = False)
    return True

def readMatrix(dir):

    df = pd.read_csv(dir)
    coo = sps.coo_matrix((df["data"].values,
                                  (df.iloc[:,0].values,df.iloc[:,1].values)),
                                  shape=(int(df.columns[0]),int(df.columns[1])))
    return coo.tocsr()

