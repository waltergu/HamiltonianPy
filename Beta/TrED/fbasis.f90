subroutine trbasis(table,dt,nt,nstate,seqs,maps,translations,signs,nbasis,ntable)
    implicit none
    integer(4),intent(in) :: ntable
    integer(8),intent(in) :: table(ntable)
    integer(4),intent(in) :: dt,nt,nstate
    integer(8),intent(out) :: seqs(ntable),maps(ntable),translations(ntable),signs(ntable)
    integer(4),intent(out) :: nbasis
    integer(4) :: i,j,seq
    integer(8) :: basis
    logical(4) :: mask(ntable)
    mask=.true.
    nbasis=0
    do i=1,ntable
        if(mask(i)) then
            nbasis=nbasis+1
            seqs(nbasis)=i-1
            do j=1,nt
                basis=ishftc(table(i),j*dt,nstate)
                seq=searchsorted(table,basis,ntable)
                maps(seq)=nbasis-1
                translations(seq)=j
                signs(seq)=(-1)**(nbts(table(i),0,nstate-j*dt-1)*nbts(table(i),nstate-j*dt,nstate-1))
                if(basis==table(i)) exit
                mask(seq)=.false.
            end do
        end if
    end do
    contains
        function searchsorted(table,basis,ntable) result(seq)
            implicit none
            integer(4) :: seq
            integer(4),intent(in) :: ntable
            integer(8),intent(in) :: table(ntable)
            integer(8),intent(in) :: basis
            integer(4) :: lb,ub
            lb=1
            ub=ntable+1
            seq=(lb+ub)/2
            do
                if(table(seq)>basis) then
                    ub=seq
                else
                    lb=seq
                end if
                seq=(lb+ub)/2
                if(table(seq)==basis) exit
            end do
        end function searchsorted
        function nbts(basis,start,end) result(num)
            implicit none
            integer(4) :: num
            integer(8),intent(in) :: basis
            integer(4),intent(in) :: start,end
            integer(4) :: i
            num=0
            do i=start,end
                if(btest(basis,i)) num=num+1
            end do
        end function nbts
end subroutine trbasis

subroutine trmasks(seqs,translations,signs,nk,masks,nbasis,ntable)
    implicit none
    integer(4),intent(in) :: nk,nbasis,ntable
    integer(4),intent(in) :: seqs(nbasis),translations(ntable),signs(ntable)
    integer(4),intent(out) :: masks(nk,nbasis)
    integer(4) :: i,j,k,count
    complex(8) :: delta,sum
    real(8) :: norm
    delta=exp((0.0,-1.0)*2*3.1415926536/nk)
    do i=0,nk-1
        count=0
        do j=1,nbasis
            sum=0.0
            do k=0,nk/translations(seqs(j)+1)-1
                sum=sum+signs(seqs(j)+1)**k*delta**(i*k*translations(seqs(j)+1))
            end do
            norm=abs(sum)
            if(norm>0.0001) then
                if (abs(norm-nk/translations(seqs(j)+1))>0.0001) call exit(1)
                masks(i+1,j)=count
                count=count+1
            else
                masks(i+1,j)=-1
            end if
        end do
    end do
end subroutine trmasks
