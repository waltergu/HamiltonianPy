from mpi4py import MPI
import numpy as np
import mkl,time

__all__=[]

mkl.set_num_threads(1)
comm=MPI.Comm.Get_parent()
vecs,lczs,indices=comm.recv(source=0,tag=0)
data=[]
for index,lanczos in zip(indices,lczs):
    stime=time.time()
    Q=np.zeros((vecs.shape[0],lanczos.maxiter),dtype=vecs.dtype)
    while lanczos.niter<lanczos.maxiter and not lanczos.stop:
        lanczos.iter()
        Q[:,lanczos.niter-1]=vecs.dot(lanczos.vectors[lanczos.niter-1])
    data.append((index,(lanczos._T_,lanczos.P,lanczos.niter),Q))
    etime=time.time()
    comm.send((index,etime-stime),dest=0,tag=1)
comm.send(data,dest=0,tag=0)
comm.Disconnect()
