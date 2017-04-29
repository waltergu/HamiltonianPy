subroutine fkron_r4(d1,inds1,indp1,rcs1,d2,inds2,indp2,rcs2,nnz,slices,d,inds,indp,shp1,shp2,nd1,nd2,nrc)
    implicit none
    integer,intent(in) :: nnz,shp1,shp2,nd1,nd2,nrc
    real(4),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(shp1+1),indp2(shp2+1)
    integer,intent(in) :: rcs1(nrc),rcs2(nrc)
    integer,intent(in) :: slices(shp1*shp2)
    real(4),intent(out) :: d(nnz)
    integer,intent(out) :: inds(nnz)
    integer,intent(out) :: indp(nrc+1)
    integer :: rc1,rc2,i,j,k,inc,pos
    indp(1)=0
    do i=1,nrc
        rc1=rcs1(i)+1
        rc2=rcs2(i)+1
        pos=indp(i)
        inc=indp2(rc2+1)-indp2(rc2)
        do j=indp1(rc1)+1,indp1(rc1+1)
            do k=1,inc
                inds(k+pos)=slices(inds1(j)*shp2+inds2(indp2(rc2)+k)+1)
            end do
            d(pos+1:pos+inc)=d1(j)*d2(indp2(rc2)+1:indp2(rc2+1))
            pos=pos+inc
        end do
        indp(i+1)=pos
    end do
end subroutine fkron_r4

subroutine fkron_r8(d1,inds1,indp1,rcs1,d2,inds2,indp2,rcs2,nnz,slices,d,inds,indp,shp1,shp2,nd1,nd2,nrc)
    implicit none
    integer,intent(in) :: nnz,shp1,shp2,nd1,nd2,nrc
    real(8),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(shp1+1),indp2(shp2+1)
    integer,intent(in) :: rcs1(nrc),rcs2(nrc)
    integer,intent(in) :: slices(shp1*shp2)
    real(8),intent(out) :: d(nnz)
    integer,intent(out) :: inds(nnz)
    integer,intent(out) :: indp(nrc+1)
    integer :: rc1,rc2,i,j,k,inc,pos
    indp(1)=0
    do i=1,nrc
        rc1=rcs1(i)+1
        rc2=rcs2(i)+1
        pos=indp(i)
        inc=indp2(rc2+1)-indp2(rc2)
        do j=indp1(rc1)+1,indp1(rc1+1)
            do k=1,inc
                inds(k+pos)=slices(inds1(j)*shp2+inds2(indp2(rc2)+k)+1)
            end do
            d(pos+1:pos+inc)=d1(j)*d2(indp2(rc2)+1:indp2(rc2+1))
            pos=pos+inc
        end do
        indp(i+1)=pos
    end do
end subroutine fkron_r8

subroutine fkron_c4(d1,inds1,indp1,rcs1,d2,inds2,indp2,rcs2,nnz,slices,d,inds,indp,shp1,shp2,nd1,nd2,nrc)
    implicit none
    integer,intent(in) :: nnz,shp1,shp2,nd1,nd2,nrc
    complex(4),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(shp1+1),indp2(shp2+1)
    integer,intent(in) :: rcs1(nrc),rcs2(nrc)
    integer,intent(in) :: slices(shp1*shp2)
    complex(4),intent(out) :: d(nnz)
    integer,intent(out) :: inds(nnz)
    integer,intent(out) :: indp(nrc+1)
    integer :: rc1,rc2,i,j,k,inc,pos
    indp(1)=0
    do i=1,nrc
        rc1=rcs1(i)+1
        rc2=rcs2(i)+1
        pos=indp(i)
        inc=indp2(rc2+1)-indp2(rc2)
        do j=indp1(rc1)+1,indp1(rc1+1)
            do k=1,inc
                inds(k+pos)=slices(inds1(j)*shp2+inds2(indp2(rc2)+k)+1)
            end do
            d(pos+1:pos+inc)=d1(j)*d2(indp2(rc2)+1:indp2(rc2+1))
            pos=pos+inc
        end do
        indp(i+1)=pos
    end do
end subroutine fkron_c4

subroutine fkron_c8(d1,inds1,indp1,rcs1,d2,inds2,indp2,rcs2,nnz,slices,d,inds,indp,shp1,shp2,nd1,nd2,nrc)
    implicit none
    integer,intent(in) :: nnz,shp1,shp2,nd1,nd2,nrc
    complex(8),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(shp1+1),indp2(shp2+1)
    integer,intent(in) :: rcs1(nrc),rcs2(nrc)
    integer,intent(in) :: slices(shp1*shp2)
    complex(8),intent(out) :: d(nnz)
    integer,intent(out) :: inds(nnz)
    integer,intent(out) :: indp(nrc+1)
    integer :: rc1,rc2,i,j,k,inc,pos
    indp(1)=0
    do i=1,nrc
        rc1=rcs1(i)+1
        rc2=rcs2(i)+1
        pos=indp(i)
        inc=indp2(rc2+1)-indp2(rc2)
        do j=indp1(rc1)+1,indp1(rc1+1)
            do k=1,inc
                inds(k+pos)=slices(inds1(j)*shp2+inds2(indp2(rc2)+k)+1)
            end do
            d(pos+1:pos+inc)=d1(j)*d2(indp2(rc2)+1:indp2(rc2+1))
            pos=pos+inc
        end do
        indp(i+1)=pos
    end do
end subroutine fkron_c8
