subroutine fkron(m1,m2,pairs,antipermutation,tol,data,row,col,count,n1,n2,npairs,ndata)
    integer,intent(in) :: n1,n2,npairs,ndata
    real(8),intent(in),dimension(n1,n1) :: m1
    real(8),intent(in),dimension(n2,n2) :: m2
    integer,intent(in),dimension(4,npairs) :: pairs
    integer,intent(in),dimension(n1*n2) :: antipermutation
    real(8),intent(in) :: tol
    real(8),intent(out),dimension(ndata) :: data
    integer,intent(out),dimension(ndata) :: row,col
    integer,intent(out) :: count
    integer :: i,j,k,l,m,n
    real(8) :: temp
    count=1
    do i=1,npairs
        do j=1,npairs
            do k=pairs(1,i),pairs(2,i)
                do l=pairs(3,i),pairs(4,i)
                    do m=pairs(1,j),pairs(2,j)
                        do n=pairs(3,j),pairs(4,j)
                            temp=m1(k,m)*m2(l,n)
                            if(abs(temp)>tol)then
                                data(count)=temp
                                row(count)=antipermutation((k-1)*n2+l)
                                col(count)=antipermutation((m-1)*n2+n)
                                count=count+1
                            end if
                        end do
                    end do
                end do
            end do
        end do
    end do
end subroutine fkron
