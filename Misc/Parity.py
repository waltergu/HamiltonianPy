'''
The parity of a permutation.
'''

__all__=['parity']

def parity(permutation):
    '''
    Determine the parity of a permutation.
    Parameters:
        permutation: list of integer
            A permutation of integers from 0 to N-1.
    Returns: -1 or +1
        -1 for odd permutation, and
        +1 for even permutation.
    '''
    result=1
    for i in xrange(len(permutation)-1):
        if permutation[i]!=i:
            result*=-1
            pos=min(xrange(i,len(permutation)),key=permutation.__getitem__)
            permutation[i],permutation[pos]=permutation[pos],permutation[i]
    return result
