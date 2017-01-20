subroutine gf_contract_4(k,mgf,seqs,coords,gf,ndim,ncgf,ngf,nclmap)
    integer,intent(in) :: ndim,ncgf,ngf,nclmap
    real(4),intent(in),dimension(ndim) :: k
    complex(4),intent(in),dimension(ncgf,ncgf) :: mgf
    integer,intent(in),dimension(ngf,nclmap) :: seqs
    real(4),intent(in),dimension(ngf,nclmap,ndim) :: coords
    complex(4),intent(out),dimension(ngf,ngf) :: gf
    integer :: i,j,m,n,h
    real(4),dimension(ndim) :: coords_buff
    gf=(0.0_4,0.0_4)
    do i=1,ngf
        do m=1,nclmap
            do j=1,ngf
                do n=1,nclmap
                    do h=1,ndim
                      coords_buff(h)=coords(j,n,h)-coords(i,m,h)
                    end do
                    gf(i,j)=gf(i,j)+mgf(seqs(i,m),seqs(j,n))*exp((0.0_8,1.0_8)*dot_product(k,coords_buff))
                end do
            end do
        end do
    end do
end subroutine gf_contract_4

subroutine gf_contract_8(k,mgf,seqs,coords,gf,ndim,ncgf,ngf,nclmap)
    integer,intent(in) :: ndim,ncgf,ngf,nclmap
    real(8),intent(in),dimension(ndim) :: k
    complex(8),intent(in),dimension(ncgf,ncgf) :: mgf
    integer,intent(in),dimension(ngf,nclmap) :: seqs
    real(8),intent(in),dimension(ngf,nclmap,ndim) :: coords
    complex(8),intent(out),dimension(ngf,ngf) :: gf
    integer :: i,j,m,n,h
    real(8),dimension(ndim) :: coords_buff
    gf=(0.0_8,0.0_8)
    do i=1,ngf
        do m=1,nclmap
            do j=1,ngf
                do n=1,nclmap
                    do h=1,ndim
                      coords_buff(h)=coords(j,n,h)-coords(i,m,h)
                    end do
                    gf(i,j)=gf(i,j)+mgf(seqs(i,m),seqs(j,n))*exp((0.0_8,1.0_8)*dot_product(k,coords_buff))
                end do
            end do
        end do
    end do
end subroutine gf_contract_8
