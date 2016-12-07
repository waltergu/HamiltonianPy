subroutine fkron_csr_r4(d1,inds1,indp1,shp1,d2,inds2,indp2,shp2,rs,d,inds,indp,shp,nd1,n1,nd2,n2,nr)
    implicit none
    integer,intent(in) :: nd1,n1,nd2,n2,nr
    real(4),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(n1),indp2(n2)
    integer,intent(in) :: shp1(2),shp2(2)
    integer,intent(in) :: rs(nr)
    real(4),intent(out) :: d(nd1*nd2)
    integer,intent(out) :: inds(nd1*nd2)
    integer,intent(out) :: indp(nr+1)
    integer,intent(out) :: shp(2)
    integer :: r1,r2,i,j,count,total
    indp(1)=0
    do i=1,nr
        r1=rs(i)/shp2(1)+1
        r2=MOD(rs(i),shp2(1))+1
        total=0
        do j=indp1(r1)+1,indp1(r1+1)
            count=indp2(r2+1)-indp2(r2)
            inds(indp(i)+total+1:indp(i)+total+count)=inds1(j)*shp2(2)+inds2(indp2(r2)+1:indp2(r2+1))
            d(indp(i)+total+1:indp(i)+total+count)=d1(j)*d2(indp2(r2)+1:indp2(r2+1))
            total=total+count
        end do
        indp(i+1)=indp(i)+total
    end do
    shp(1)=nr
    shp(2)=shp1(2)*shp2(2)
end subroutine fkron_csr_r4

subroutine fkron_csr_r8(d1,inds1,indp1,shp1,d2,inds2,indp2,shp2,rs,d,inds,indp,shp,nd1,n1,nd2,n2,nr)
    implicit none
    integer,intent(in) :: nd1,n1,nd2,n2,nr
    real(8),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(n1),indp2(n2)
    integer,intent(in) :: shp1(2),shp2(2)
    integer,intent(in) :: rs(nr)
    real(8),intent(out) :: d(nd1*nd2)
    integer,intent(out) :: inds(nd1*nd2)
    integer,intent(out) :: indp(nr+1)
    integer,intent(out) :: shp(2)
    integer :: r1,r2,i,j,count,total
    indp(1)=0
    do i=1,nr
        r1=rs(i)/shp2(1)+1
        r2=MOD(rs(i),shp2(1))+1
        total=0
        do j=indp1(r1)+1,indp1(r1+1)
            count=indp2(r2+1)-indp2(r2)
            inds(indp(i)+total+1:indp(i)+total+count)=inds1(j)*shp2(2)+inds2(indp2(r2)+1:indp2(r2+1))
            d(indp(i)+total+1:indp(i)+total+count)=d1(j)*d2(indp2(r2)+1:indp2(r2+1))
            total=total+count
        end do
        indp(i+1)=indp(i)+total
    end do
    shp(1)=nr
    shp(2)=shp1(2)*shp2(2)
end subroutine fkron_csr_r8

subroutine fkron_csr_c4(d1,inds1,indp1,shp1,d2,inds2,indp2,shp2,rs,d,inds,indp,shp,nd1,n1,nd2,n2,nr)
    implicit none
    integer,intent(in) :: nd1,n1,nd2,n2,nr
    complex(4),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(n1),indp2(n2)
    integer,intent(in) :: shp1(2),shp2(2)
    integer,intent(in) :: rs(nr)
    complex(4),intent(out) :: d(nd1*nd2)
    integer,intent(out) :: inds(nd1*nd2)
    integer,intent(out) :: indp(nr+1)
    integer,intent(out) :: shp(2)
    integer :: r1,r2,i,j,count,total
    indp(1)=0
    do i=1,nr
        r1=rs(i)/shp2(1)+1
        r2=MOD(rs(i),shp2(1))+1
        total=0
        do j=indp1(r1)+1,indp1(r1+1)
            count=indp2(r2+1)-indp2(r2)
            inds(indp(i)+total+1:indp(i)+total+count)=inds1(j)*shp2(2)+inds2(indp2(r2)+1:indp2(r2+1))
            d(indp(i)+total+1:indp(i)+total+count)=d1(j)*d2(indp2(r2)+1:indp2(r2+1))
            total=total+count
        end do
        indp(i+1)=indp(i)+total
    end do
    shp(1)=nr
    shp(2)=shp1(2)*shp2(2)
end subroutine fkron_csr_c4

subroutine fkron_csr_c8(d1,inds1,indp1,shp1,d2,inds2,indp2,shp2,rs,d,inds,indp,shp,nd1,n1,nd2,n2,nr)
    implicit none
    integer,intent(in) :: nd1,n1,nd2,n2,nr
    complex(8),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(n1),indp2(n2)
    integer,intent(in) :: shp1(2),shp2(2)
    integer,intent(in) :: rs(nr)
    complex(8),intent(out) :: d(nd1*nd2)
    integer,intent(out) :: inds(nd1*nd2)
    integer,intent(out) :: indp(nr+1)
    integer,intent(out) :: shp(2)
    integer :: r1,r2,i,j,count,total
    indp(1)=0
    do i=1,nr
        r1=rs(i)/shp2(1)+1
        r2=MOD(rs(i),shp2(1))+1
        total=0
        do j=indp1(r1)+1,indp1(r1+1)
            count=indp2(r2+1)-indp2(r2)
            inds(indp(i)+total+1:indp(i)+total+count)=inds1(j)*shp2(2)+inds2(indp2(r2)+1:indp2(r2+1))
            d(indp(i)+total+1:indp(i)+total+count)=d1(j)*d2(indp2(r2)+1:indp2(r2+1))
            total=total+count
        end do
        indp(i+1)=indp(i)+total
    end do
    shp(1)=nr
    shp(2)=shp1(2)*shp2(2)
end subroutine fkron_csr_c8

subroutine fkron_csc_r4(d1,inds1,indp1,shp1,d2,inds2,indp2,shp2,cs,d,inds,indp,shp,nd1,n1,nd2,n2,nc)
    implicit none
    integer,intent(in) :: nd1,n1,nd2,n2,nc
    real(4),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(n1),indp2(n2)
    integer,intent(in) :: shp1(2),shp2(2)
    integer,intent(in) :: cs(nc)
    real(4),intent(out) :: d(nd1*nd2)
    integer,intent(out) :: inds(nd1*nd2)
    integer,intent(out) :: indp(nc+1)
    integer,intent(out) :: shp(2)
    integer :: c1,c2,i,j,count,total
    indp(1)=0
    do i=1,nc
        c1=cs(i)/shp2(2)+1
        c2=MOD(cs(i),shp2(2))+1
        total=0
        do j=indp1(c1)+1,indp1(c1+1)
            count=indp2(c2+1)-indp2(c2)
            inds(indp(i)+total+1:indp(i)+total+count)=inds1(j)*shp2(1)+inds2(indp2(c2)+1:indp2(c2+1))
            d(indp(i)+total+1:indp(i)+total+count)=d1(j)*d2(indp2(c2)+1:indp2(c2+1))
            total=total+count
        end do
        indp(i+1)=indp(i)+total
    end do
    shp(1)=shp1(1)*shp2(1)
    shp(2)=nc
end subroutine fkron_csc_r4

subroutine fkron_csc_r8(d1,inds1,indp1,shp1,d2,inds2,indp2,shp2,cs,d,inds,indp,shp,nd1,n1,nd2,n2,nc)
    implicit none
    integer,intent(in) :: nd1,n1,nd2,n2,nc
    real(8),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(n1),indp2(n2)
    integer,intent(in) :: shp1(2),shp2(2)
    integer,intent(in) :: cs(nc)
    real(8),intent(out) :: d(nd1*nd2)
    integer,intent(out) :: inds(nd1*nd2)
    integer,intent(out) :: indp(nc+1)
    integer,intent(out) :: shp(2)
    integer :: c1,c2,i,j,count,total
    indp(1)=0
    do i=1,nc
        c1=cs(i)/shp2(2)+1
        c2=MOD(cs(i),shp2(2))+1
        total=0
        do j=indp1(c1)+1,indp1(c1+1)
            count=indp2(c2+1)-indp2(c2)
            inds(indp(i)+total+1:indp(i)+total+count)=inds1(j)*shp2(1)+inds2(indp2(c2)+1:indp2(c2+1))
            d(indp(i)+total+1:indp(i)+total+count)=d1(j)*d2(indp2(c2)+1:indp2(c2+1))
            total=total+count
        end do
        indp(i+1)=indp(i)+total
    end do
    shp(1)=shp1(1)*shp2(1)
    shp(2)=nc
end subroutine fkron_csc_r8

subroutine fkron_csc_c4(d1,inds1,indp1,shp1,d2,inds2,indp2,shp2,cs,d,inds,indp,shp,nd1,n1,nd2,n2,nc)
    implicit none
    integer,intent(in) :: nd1,n1,nd2,n2,nc
    complex(4),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(n1),indp2(n2)
    integer,intent(in) :: shp1(2),shp2(2)
    integer,intent(in) :: cs(nc)
    complex(4),intent(out) :: d(nd1*nd2)
    integer,intent(out) :: inds(nd1*nd2)
    integer,intent(out) :: indp(nc+1)
    integer,intent(out) :: shp(2)
    integer :: c1,c2,i,j,count,total
    indp(1)=0
    do i=1,nc
        c1=cs(i)/shp2(2)+1
        c2=MOD(cs(i),shp2(2))+1
        total=0
        do j=indp1(c1)+1,indp1(c1+1)
            count=indp2(c2+1)-indp2(c2)
            inds(indp(i)+total+1:indp(i)+total+count)=inds1(j)*shp2(1)+inds2(indp2(c2)+1:indp2(c2+1))
            d(indp(i)+total+1:indp(i)+total+count)=d1(j)*d2(indp2(c2)+1:indp2(c2+1))
            total=total+count
        end do
        indp(i+1)=indp(i)+total
    end do
    shp(1)=shp1(1)*shp2(1)
    shp(2)=nc
end subroutine fkron_csc_c4

subroutine fkron_csc_c8(d1,inds1,indp1,shp1,d2,inds2,indp2,shp2,cs,d,inds,indp,shp,nd1,n1,nd2,n2,nc)
    implicit none
    integer,intent(in) :: nd1,n1,nd2,n2,nc
    complex(8),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(n1),indp2(n2)
    integer,intent(in) :: shp1(2),shp2(2)
    integer,intent(in) :: cs(nc)
    complex(8),intent(out) :: d(nd1*nd2)
    integer,intent(out) :: inds(nd1*nd2)
    integer,intent(out) :: indp(nc+1)
    integer,intent(out) :: shp(2)
    integer :: c1,c2,i,j,count,total
    indp(1)=0
    do i=1,nc
        c1=cs(i)/shp2(2)+1
        c2=MOD(cs(i),shp2(2))+1
        total=0
        do j=indp1(c1)+1,indp1(c1+1)
            count=indp2(c2+1)-indp2(c2)
            inds(indp(i)+total+1:indp(i)+total+count)=inds1(j)*shp2(1)+inds2(indp2(c2)+1:indp2(c2+1))
            d(indp(i)+total+1:indp(i)+total+count)=d1(j)*d2(indp2(c2)+1:indp2(c2+1))
            total=total+count
        end do
        indp(i+1)=indp(i)+total
    end do
    shp(1)=shp1(1)*shp2(1)
    shp(2)=nc
end subroutine fkron_csc_c8
