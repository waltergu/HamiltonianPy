subroutine fpermutation(indptr,counts,fp,fn,n)
    integer(8),intent(in) :: fn,n
    integer(8),intent(in) :: indptr(n),counts(n)
    integer(8),intent(out) :: fp(fn)
    integer(8) :: pos,i,j
    pos=0
    do i=1,n
        do j=1,counts(i)
            fp(pos+j)=indptr(i)+j-1
        end do
        pos=pos+counts(i)
    end do
end subroutine fpermutation
