subroutine fkron_r4(mode,d1,inds1,indp1,shp1,rcs1,d2,inds2,indp2,shp2,rcs2,nnz,d,inds,indp,shp,nd1,n1,nd2,n2,nrc)
    implicit none
    character(1),intent(in) :: mode
    integer,intent(in) :: nnz,nd1,n1,nd2,n2,nrc
    real(4),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(n1),indp2(n2)
    integer,intent(in) :: shp1(2),shp2(2)
    integer,intent(in) :: rcs1(nrc),rcs2(nrc)
    real(4),intent(out) :: d(nnz)
    integer,intent(out) :: inds(nnz)
    integer,intent(out) :: indp(nrc+1)
    integer,intent(out) :: shp(2)
    integer :: rc1,rc2,i,j,inc,rc,pos
    select case(mode)
        case('C');rc=shp2(1);shp(1)=shp1(1)*shp2(1);shp(2)=nrc
        case('R');rc=shp2(2);shp(1)=nrc;shp(2)=shp1(2)*shp2(2)
    end select
    indp(1)=0
    do i=1,nrc
        rc1=rcs1(i)+1
        rc2=rcs2(i)+1
        pos=indp(i)
        inc=indp2(rc2+1)-indp2(rc2)
        do j=indp1(rc1)+1,indp1(rc1+1)
            inds(pos+1:pos+inc)=inds1(j)*rc+inds2(indp2(rc2)+1:indp2(rc2+1))
            d(pos+1:pos+inc)=d1(j)*d2(indp2(rc2)+1:indp2(rc2+1))
            pos=pos+inc
        end do
        indp(i+1)=pos
    end do
end subroutine fkron_r4

subroutine fkron_r8(mode,d1,inds1,indp1,shp1,rcs1,d2,inds2,indp2,shp2,rcs2,nnz,d,inds,indp,shp,nd1,n1,nd2,n2,nrc)
    implicit none
    character(1),intent(in) :: mode
    integer,intent(in) :: nnz,nd1,n1,nd2,n2,nrc
    real(8),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(n1),indp2(n2)
    integer,intent(in) :: shp1(2),shp2(2)
    integer,intent(in) :: rcs1(nrc),rcs2(nrc)
    real(8),intent(out) :: d(nnz)
    integer,intent(out) :: inds(nnz)
    integer,intent(out) :: indp(nrc+1)
    integer,intent(out) :: shp(2)
    integer :: rc1,rc2,i,j,inc,rc,pos
    select case(mode)
        case('C');rc=shp2(1);shp(1)=shp1(1)*shp2(1);shp(2)=nrc
        case('R');rc=shp2(2);shp(1)=nrc;shp(2)=shp1(2)*shp2(2)
    end select
    indp(1)=0
    do i=1,nrc
        rc1=rcs1(i)+1
        rc2=rcs2(i)+1
        pos=indp(i)
        inc=indp2(rc2+1)-indp2(rc2)
        do j=indp1(rc1)+1,indp1(rc1+1)
            inds(pos+1:pos+inc)=inds1(j)*rc+inds2(indp2(rc2)+1:indp2(rc2+1))
            d(pos+1:pos+inc)=d1(j)*d2(indp2(rc2)+1:indp2(rc2+1))
            pos=pos+inc
        end do
        indp(i+1)=pos
    end do
end subroutine fkron_r8

subroutine fkron_c4(mode,d1,inds1,indp1,shp1,rcs1,d2,inds2,indp2,shp2,rcs2,nnz,d,inds,indp,shp,nd1,n1,nd2,n2,nrc)
    implicit none
    character(1),intent(in) :: mode
    integer,intent(in) :: nnz,nd1,n1,nd2,n2,nrc
    complex(4),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(n1),indp2(n2)
    integer,intent(in) :: shp1(2),shp2(2)
    integer,intent(in) :: rcs1(nrc),rcs2(nrc)
    complex(4),intent(out) :: d(nnz)
    integer,intent(out) :: inds(nnz)
    integer,intent(out) :: indp(nrc+1)
    integer,intent(out) :: shp(2)
    integer :: rc1,rc2,i,j,inc,rc,pos
    select case(mode)
        case('C');rc=shp2(1);shp(1)=shp1(1)*shp2(1);shp(2)=nrc
        case('R');rc=shp2(2);shp(1)=nrc;shp(2)=shp1(2)*shp2(2)
    end select
    indp(1)=0
    do i=1,nrc
        rc1=rcs1(i)+1
        rc2=rcs2(i)+1
        pos=indp(i)
        inc=indp2(rc2+1)-indp2(rc2)
        do j=indp1(rc1)+1,indp1(rc1+1)
            inds(pos+1:pos+inc)=inds1(j)*rc+inds2(indp2(rc2)+1:indp2(rc2+1))
            d(pos+1:pos+inc)=d1(j)*d2(indp2(rc2)+1:indp2(rc2+1))
            pos=pos+inc
        end do
        indp(i+1)=pos
    end do
end subroutine fkron_c4

subroutine fkron_c8(mode,d1,inds1,indp1,shp1,rcs1,d2,inds2,indp2,shp2,rcs2,nnz,d,inds,indp,shp,nd1,n1,nd2,n2,nrc)
    implicit none
    character(1),intent(in) :: mode
    integer,intent(in) :: nnz,nd1,n1,nd2,n2,nrc
    complex(8),intent(in) :: d1(nd1),d2(nd2)
    integer,intent(in) :: inds1(nd1),inds2(nd2)
    integer,intent(in) :: indp1(n1),indp2(n2)
    integer,intent(in) :: shp1(2),shp2(2)
    integer,intent(in) :: rcs1(nrc),rcs2(nrc)
    complex(8),intent(out) :: d(nnz)
    integer,intent(out) :: inds(nnz)
    integer,intent(out) :: indp(nrc+1)
    integer,intent(out) :: shp(2)
    integer :: rc1,rc2,i,j,inc,rc,pos
    select case(mode)
        case('C');rc=shp2(1);shp(1)=shp1(1)*shp2(1);shp(2)=nrc
        case('R');rc=shp2(2);shp(1)=nrc;shp(2)=shp1(2)*shp2(2)
    end select
    indp(1)=0
    do i=1,nrc
        rc1=rcs1(i)+1
        rc2=rcs2(i)+1
        pos=indp(i)
        inc=indp2(rc2+1)-indp2(rc2)
        do j=indp1(rc1)+1,indp1(rc1+1)
            inds(pos+1:pos+inc)=inds1(j)*rc+inds2(indp2(rc2)+1:indp2(rc2+1))
            d(pos+1:pos+inc)=d1(j)*d2(indp2(rc2)+1:indp2(rc2+1))
            pos=pos+inc
        end do
        indp(i+1)=pos
    end do
end subroutine fkron_c8

subroutine fkron_identity_r4(mode,md,minds,mindp,mshp,mrcs,idn,ircs,nnz,d,inds,indp,shp,mnd,mn,nrc)
    implicit none
    character(2) :: mode
    integer(4),intent(in) :: idn,nnz,mnd,mn,nrc
    real(4),intent(in) :: md(mnd)
    integer(4),intent(in) :: minds(mnd)
    integer(4),intent(in) :: mindp(mn)
    integer(4),intent(in) :: mshp(2)
    integer(4),intent(in) :: mrcs(nrc),ircs(nrc)
    real(4),intent(out) :: d(nnz)
    integer(4),intent(out) :: inds(nnz)
    integer(4),intent(out) :: indp(nrc+1)
    integer(4),intent(out) :: shp(2)
    integer :: i,mrc,rc
    select case(mode(1:1))
        case('L')
            select case(mode(2:2))
                case('C');rc=mshp(1);shp(1)=idn*mshp(1);shp(2)=nrc
                case('R');rc=mshp(2);shp(1)=nrc;shp(2)=idn*mshp(2)
            end select
            indp(1)=0
            do i=1,nrc
                mrc=mrcs(i)+1
                indp(i+1)=indp(i)+mindp(mrc+1)-mindp(mrc)
                inds(indp(i)+1:indp(i+1))=ircs(i)*rc+minds(mindp(mrc)+1:mindp(mrc+1))
                d(indp(i)+1:indp(i+1))=md(mindp(mrc)+1:mindp(mrc+1))
            end do
        case('R')
            select case(mode(2:2))
                case('C');shp(1)=mshp(1)*idn;shp(2)=nrc
                case('R');shp(1)=nrc;shp(2)=mshp(2)*idn
            end select
            indp(1)=0
            do i=1,nrc
                mrc=mrcs(i)+1
                indp(i+1)=indp(i)+mindp(mrc+1)-mindp(mrc)
                inds(indp(i)+1:indp(i+1))=minds(mindp(mrc)+1:mindp(mrc+1))*idn+ircs(i)
                d(indp(i)+1:indp(i+1))=md(mindp(mrc)+1:mindp(mrc+1))
            end do
    end select
end subroutine fkron_identity_r4

subroutine fkron_identity_r8(mode,md,minds,mindp,mshp,mrcs,idn,ircs,nnz,d,inds,indp,shp,mnd,mn,nrc)
    implicit none
    character(2) :: mode
    integer(4),intent(in) :: idn,nnz,mnd,mn,nrc
    real(8),intent(in) :: md(mnd)
    integer(4),intent(in) :: minds(mnd)
    integer(4),intent(in) :: mindp(mn)
    integer(4),intent(in) :: mshp(2)
    integer(4),intent(in) :: mrcs(nrc),ircs(nrc)
    real(8),intent(out) :: d(nnz)
    integer(4),intent(out) :: inds(nnz)
    integer(4),intent(out) :: indp(nrc+1)
    integer(4),intent(out) :: shp(2)
    integer :: i,mrc,rc
    select case(mode(1:1))
        case('L')
            select case(mode(2:2))
                case('C');rc=mshp(1);shp(1)=idn*mshp(1);shp(2)=nrc
                case('R');rc=mshp(2);shp(1)=nrc;shp(2)=idn*mshp(2)
            end select
            indp(1)=0
            do i=1,nrc
                mrc=mrcs(i)+1
                indp(i+1)=indp(i)+mindp(mrc+1)-mindp(mrc)
                inds(indp(i)+1:indp(i+1))=ircs(i)*rc+minds(mindp(mrc)+1:mindp(mrc+1))
                d(indp(i)+1:indp(i+1))=md(mindp(mrc)+1:mindp(mrc+1))
            end do
        case('R')
            select case(mode(2:2))
                case('C');shp(1)=mshp(1)*idn;shp(2)=nrc
                case('R');shp(1)=nrc;shp(2)=mshp(2)*idn
            end select
            indp(1)=0
            do i=1,nrc
                mrc=mrcs(i)+1
                indp(i+1)=indp(i)+mindp(mrc+1)-mindp(mrc)
                inds(indp(i)+1:indp(i+1))=minds(mindp(mrc)+1:mindp(mrc+1))*idn+ircs(i)
                d(indp(i)+1:indp(i+1))=md(mindp(mrc)+1:mindp(mrc+1))
            end do
    end select
end subroutine fkron_identity_r8

subroutine fkron_identity_c4(mode,md,minds,mindp,mshp,mrcs,idn,ircs,nnz,d,inds,indp,shp,mnd,mn,nrc)
    implicit none
    character(2) :: mode
    integer(4),intent(in) :: idn,nnz,mnd,mn,nrc
    complex(4),intent(in) :: md(mnd)
    integer(4),intent(in) :: minds(mnd)
    integer(4),intent(in) :: mindp(mn)
    integer(4),intent(in) :: mshp(2)
    integer(4),intent(in) :: mrcs(nrc),ircs(nrc)
    complex(4),intent(out) :: d(nnz)
    integer(4),intent(out) :: inds(nnz)
    integer(4),intent(out) :: indp(nrc+1)
    integer(4),intent(out) :: shp(2)
    integer :: i,mrc,rc
    select case(mode(1:1))
        case('L')
            select case(mode(2:2))
                case('C');rc=mshp(1);shp(1)=idn*mshp(1);shp(2)=nrc
                case('R');rc=mshp(2);shp(1)=nrc;shp(2)=idn*mshp(2)
            end select
            indp(1)=0
            do i=1,nrc
                mrc=mrcs(i)+1
                indp(i+1)=indp(i)+mindp(mrc+1)-mindp(mrc)
                inds(indp(i)+1:indp(i+1))=ircs(i)*rc+minds(mindp(mrc)+1:mindp(mrc+1))
                d(indp(i)+1:indp(i+1))=md(mindp(mrc)+1:mindp(mrc+1))
            end do
        case('R')
            select case(mode(2:2))
                case('C');shp(1)=mshp(1)*idn;shp(2)=nrc
                case('R');shp(1)=nrc;shp(2)=mshp(2)*idn
            end select
            indp(1)=0
            do i=1,nrc
                mrc=mrcs(i)+1
                indp(i+1)=indp(i)+mindp(mrc+1)-mindp(mrc)
                inds(indp(i)+1:indp(i+1))=minds(mindp(mrc)+1:mindp(mrc+1))*idn+ircs(i)
                d(indp(i)+1:indp(i+1))=md(mindp(mrc)+1:mindp(mrc+1))
            end do
    end select
end subroutine fkron_identity_c4

subroutine fkron_identity_c8(mode,md,minds,mindp,mshp,mrcs,idn,ircs,nnz,d,inds,indp,shp,mnd,mn,nrc)
    implicit none
    character(2) :: mode
    integer(4),intent(in) :: idn,nnz,mnd,mn,nrc
    complex(8),intent(in) :: md(mnd)
    integer(4),intent(in) :: minds(mnd)
    integer(4),intent(in) :: mindp(mn)
    integer(4),intent(in) :: mshp(2)
    integer(4),intent(in) :: mrcs(nrc),ircs(nrc)
    complex(8),intent(out) :: d(nnz)
    integer(4),intent(out) :: inds(nnz)
    integer(4),intent(out) :: indp(nrc+1)
    integer(4),intent(out) :: shp(2)
    integer :: i,mrc,rc
    select case(mode(1:1))
        case('L')
            select case(mode(2:2))
                case('C');rc=mshp(1);shp(1)=idn*mshp(1);shp(2)=nrc
                case('R');rc=mshp(2);shp(1)=nrc;shp(2)=idn*mshp(2)
            end select
            indp(1)=0
            do i=1,nrc
                mrc=mrcs(i)+1
                indp(i+1)=indp(i)+mindp(mrc+1)-mindp(mrc)
                inds(indp(i)+1:indp(i+1))=ircs(i)*rc+minds(mindp(mrc)+1:mindp(mrc+1))
                d(indp(i)+1:indp(i+1))=md(mindp(mrc)+1:mindp(mrc+1))
            end do
        case('R')
            select case(mode(2:2))
                case('C');shp(1)=mshp(1)*idn;shp(2)=nrc
                case('R');shp(1)=nrc;shp(2)=mshp(2)*idn
            end select
            indp(1)=0
            do i=1,nrc
                mrc=mrcs(i)+1
                indp(i+1)=indp(i)+mindp(mrc+1)-mindp(mrc)
                inds(indp(i)+1:indp(i+1))=minds(mindp(mrc)+1:mindp(mrc+1))*idn+ircs(i)
                d(indp(i)+1:indp(i+1))=md(mindp(mrc)+1:mindp(mrc+1))
            end do
    end select
end subroutine fkron_identity_c8
