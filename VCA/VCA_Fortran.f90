subroutine gf_contract(k,gf_buff,seqs,coords,gf_vca,ndim,ngf,ngf_vca,nclmap)
    integer,intent(in) :: ndim,ngf,ngf_vca,nclmap
    real(8),intent(in),dimension(ndim) :: k
    complex(8),intent(in),dimension(ngf,ngf) :: gf_buff
    integer,intent(in),dimension(ngf_vca,nclmap) :: seqs
    real(8),intent(in),dimension(ngf_vca,nclmap,ndim) :: coords
    complex(8),intent(out),dimension(ngf_vca,ngf_vca) :: gf_vca
    integer :: i,j,m,n,h
    real(8),dimension(ndim) :: coords_buff
    gf_vca=(0.0_8,0.0_8)
    do i=1,ngf_vca
        do m=1,nclmap
            do j=1,ngf_vca
                do n=1,nclmap
                    do h=1,ndim
                      coords_buff(h)=coords(j,n,h)-coords(i,m,h)
                    end do
                    gf_vca(i,j)=gf_vca(i,j)+gf_buff(seqs(i,m),seqs(j,n))*exp((0.0_8,1.0_8)*dot_product(k,coords_buff))
                end do
            end do
        end do
    end do
end subroutine gf_contract
