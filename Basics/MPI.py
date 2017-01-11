'''
Wrapper for the parallel running of functions using the mpi4p, including:
1) functions: mpirun
'''

from mpi4py import MPI

__all__=['mpirun']

def mpirun(f,arguments,bcast=True):
    '''
    Wrapper for the parallel running of f using the mpi4py.
    Parameters:
        f: callable
            The function to be parallelly runned using the mpi4py.
        arguments: list of tuple
            The list of arguments passed to the function f.
        bcast: True or False
            When True, broadcast the result for all processes;
            Otherwise only the rank 0 process hold the result.
    Returns: list
        The returned values of f with respect to the arguments.
    '''
    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    rank=comm.Get_rank()
    if size>1:
        import mkl
        mkl.set_num_threads(1)
    temp=[]
    for i,argument in enumerate(arguments):
        if i%size==rank:
            temp.append(f(*argument))
    temp=comm.gather(temp,root=0)
    result=[]
    if rank==0:
        for i in xrange(len(arguments)):
            result.append(temp[i%size][i/size])
    if bcast:
        result=comm.bcast(result,root=0)
    return result
