subroutine mhubbard_r4(value,seq,permutation,u1,u2,matrix,ns,nk,nsp)
    implicit none
    integer(4),intent(in) :: ns,nk,nsp,seq
    integer(4),intent(in) :: permutation(nk)
    real(4),intent(in) :: value,u1(ns,nk,nsp),u2(ns,nk,nsp)
    real(4),intent(out) :: matrix(nk,nsp,nsp,nk,nsp,nsp)
    real(4) :: diagsum
    real(4) :: A(nsp,nsp),B(nsp,nsp)
    integer(4) :: i,j,k,l,m,n
    diagsum=sum(u2(seq,:,:)*u2(seq,:,:))
    do i=1,nk
        do j=1,nk
            do k=1,nsp
                do l=1,nsp
                    A(k,l)=u1(seq,i,k)*u1(seq,j,l)
                    if(i==j.and.k==l)then
                        B(k,l)=diagsum-u2(seq,permutation(i),k)*u2(seq,permutation(j),l)
                    else
                        B(k,l)=-u2(seq,permutation(i),k)*u2(seq,permutation(j),l)
                    end if
                end do
            end do
            do k=1,nsp
                do l=1,nsp
                    do m=1,nsp
                        do n=1,nsp
                            matrix(i,k,l,j,m,n)=A(k,m)*B(l,n)*value/nk
                        end do
                    end do
                end do
            end do
        end do
    end do
end subroutine

subroutine mhubbard_r8(value,seq,permutation,u1,u2,matrix,ns,nk,nsp)
    implicit none
    integer(4),intent(in) :: ns,nk,nsp,seq
    integer(4),intent(in) :: permutation(nk)
    real(8),intent(in) :: value,u1(ns,nk,nsp),u2(ns,nk,nsp)
    real(8),intent(out) :: matrix(nk,nsp,nsp,nk,nsp,nsp)
    real(8) :: diagsum
    real(8) :: A(nsp,nsp),B(nsp,nsp)
    integer(4) :: i,j,k,l,m,n
    diagsum=sum(u2(seq,:,:)*u2(seq,:,:))
    do i=1,nk
        do j=1,nk
            do k=1,nsp
                do l=1,nsp
                    A(k,l)=u1(seq,i,k)*u1(seq,j,l)
                    if(i==j.and.k==l)then
                        B(k,l)=diagsum-u2(seq,permutation(i),k)*u2(seq,permutation(j),l)
                    else
                        B(k,l)=-u2(seq,permutation(i),k)*u2(seq,permutation(j),l)
                    end if
                end do
            end do
            do k=1,nsp
                do l=1,nsp
                    do m=1,nsp
                        do n=1,nsp
                            matrix(i,k,l,j,m,n)=A(k,m)*B(l,n)*value/nk
                        end do
                    end do
                end do
            end do
        end do
    end do
end subroutine

subroutine mhubbard_c4(value,seq,permutation,u1,u2,matrix,ns,nk,nsp)
    implicit none
    integer(4),intent(in) :: ns,nk,nsp,seq
    integer(4),intent(in) :: permutation(nk)
    complex(4),intent(in) :: value,u1(ns,nk,nsp),u2(ns,nk,nsp)
    complex(4),intent(out) :: matrix(nk,nsp,nsp,nk,nsp,nsp)
    complex(4) :: diagsum
    complex(4) :: A(nsp,nsp),B(nsp,nsp)
    integer(4) :: i,j,k,l,m,n
    diagsum=sum(conjg(u2(seq,:,:))*u2(seq,:,:))
    do i=1,nk
        do j=1,nk
            do k=1,nsp
                do l=1,nsp
                    A(k,l)=u1(seq,i,k)*conjg(u1(seq,j,l))
                    if(i==j.and.k==l)then
                        B(k,l)=diagsum-conjg(u2(seq,permutation(i),k))*u2(seq,permutation(j),l)
                    else
                        B(k,l)=-conjg(u2(seq,permutation(i),k))*u2(seq,permutation(j),l)
                    end if
                end do
            end do
            do k=1,nsp
                do l=1,nsp
                    do m=1,nsp
                        do n=1,nsp
                            matrix(i,k,l,j,m,n)=A(k,m)*B(l,n)*value/nk
                        end do
                    end do
                end do
            end do
        end do
    end do
end subroutine

subroutine mhubbard_c8(value,seq,permutation,u1,u2,matrix,ns,nk,nsp)
    implicit none
    integer(4),intent(in) :: ns,nk,nsp,seq
    integer(4),intent(in) :: permutation(nk)
    complex(8),intent(in) :: value,u1(ns,nk,nsp),u2(ns,nk,nsp)
    complex(8),intent(out) :: matrix(nk,nsp,nsp,nk,nsp,nsp)
    complex(8) :: diagsum
    complex(8) :: A(nsp,nsp),B(nsp,nsp)
    integer(4) :: i,j,k,l,m,n
    diagsum=sum(conjg(u2(seq,:,:))*u2(seq,:,:))
    do i=1,nk
        do j=1,nk
            do k=1,nsp
                do l=1,nsp
                    A(k,l)=u1(seq,i,k)*conjg(u1(seq,j,l))
                    if(i==j.and.k==l)then
                        B(k,l)=diagsum-conjg(u2(seq,permutation(i),k))*u2(seq,permutation(j),l)
                    else
                        B(k,l)=-conjg(u2(seq,permutation(i),k))*u2(seq,permutation(j),l)
                    end if
                end do
            end do
            do k=1,nsp
                do l=1,nsp
                    do m=1,nsp
                        do n=1,nsp
                            matrix(i,k,l,j,m,n)=A(k,m)*B(l,n)*value/nk
                        end do
                    end do
                end do
            end do
        end do
    end do
end subroutine
