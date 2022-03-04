module sparse_ir
  use input_ir
  implicit none
contains
  subroutine multiply_fft_bff(va, vb, vc, n1, n2, n3, n4B, n4F)
    include "fftw3.f"
    integer, intent(in):: n1, n2, n3, n4B, n4F
    integer:: i,j,k,n
    integer:: info, lwork
    complex(8):: va(n1,n2,n3,n4B), vb(n1,n2,n3,n4F), vc(n1,n2,n3,n4F)
    complex(8), allocatable:: va_in(:,:,:,:), vb_in(:,:,:,:)
    complex(8), allocatable:: va_rm2D(:,:), vb_rm2D(:,:), vc_rm2D(:,:)
    complex(8), allocatable:: va_rt2D(:,:), vb_rt2D(:,:), vc_rt2D(:,:)
    complex(8), allocatable:: temp_bose_uln(:,:), temp_fermi_uln(:,:), temp_bose_ulx_fermi(:,:), temp_fermi_ulx_C(:,:)
    complex(8), allocatable:: tempMatrix1(:,:), tempMatrix2(:,:)
    complex(8), allocatable:: u(:,:), vt(:,:)
    complex(8), allocatable:: u_H(:,:),A_pseudo(:,:)
    complex(8), allocatable:: Sigma_u_H(:,:)
    real(8), allocatable:: s(:)
    real(8), allocatable:: ss(:)
    real(8), allocatable:: temp_fermi_ulx(:,:)
    logical, save:: initialized = .false.
    integer:: nrank, rank(3), comp
    complex(8) :: zero = (0.0d0, 0.0d0)
    complex(8) :: one = (1.0d0, 0.0d0)
    complex(8), allocatable:: work(:), rwork(:)

    nrank = 3
    rank(1:3) = (/ n1, n2, n3 /)
    allocate(va_in(n1,n2,n3,n4B))
    allocate(vb_in(n1,n2,n3,n4F))
    va_in=va
    vb_in=vb
    !vag = fft(va)
    
    call fft_k(va, n1, n2, n3, n4B)

    if (.not. allocated(va_rm2D)) allocate(va_rm2D(n4B,n1*n2*n3))
    do n=1, n4B
       do i=1,n1
          do j=1,n2
             do k=1,n3
                va_rm2D(n,(i-1)*n2*n3+(j-1)*n3+k)=va(i,j,k,n)
             end do
          end do
       end do
    end do
    
    if (.not. allocated(temp_bose_uln)) allocate(temp_bose_uln(data_num_bm,l_B+1))
    temp_bose_uln=bose_uln

    !allocate
    comp=min(data_num_bm, l_B+1)
    if (.not. allocated(work)) allocate(work(1))
    if (.not. allocated(rwork)) allocate(rwork(comp))
    if (.not. allocated(s)) allocate(s(comp))
    if (.not. allocated(u)) allocate(u(data_num_bm, data_num_bm))
    if (.not. allocated(vt)) allocate(vt(l_B+1, l_B+1))

    !workspace 
    lwork = -1
    call zgesvd('S','S',data_num_bm,l_B+1,temp_bose_uln,data_num_bm,s,u, data_num_bm,vt, l_B+1, work, lwork, rwork, info)
    
    !SVD by ?gesvd
    lwork=int(work(1))
    deallocate(work)
    allocate(work(lwork))
    call zgesvd('S','S',data_num_bm,l_B+1,temp_bose_uln,data_num_bm,s,u, data_num_bm,vt, l_B+1, work, lwork, rwork, info)
    deallocate(work)

    !inverse of singular values
    if (.not. allocated(ss)) allocate(ss(comp)) 
    do i=1,comp
      if (s(i)>=1.0d-9*maxval(s(:))) then 
       ss(i)=1.0d0/s(i)
      else
       ss(i)=s(i)
      end if
    end do

    !calculate pseudo inverse, step 1
    if (.not. allocated(u_H)) allocate(u_H(data_num_bm,data_num_bm))
    if (.not. allocated(Sigma_u_H)) allocate(Sigma_u_H(l_B+1,data_num_bm))
    u_H=transpose(conjg(u))
    do i=1,l_B+1
      Sigma_u_H(i,:)=ss(i)*u_H(i,:)
    enddo
    
    !calculate pseudo inverse, step 2
    if (.not. allocated(A_pseudo)) allocate(A_pseudo(l_B+1,data_num_bm))
    call zgemm('C', 'N', l_B+1, data_num_bm, l_B+1, one, vt, l_B+1, Sigma_u_H, l_B+1, zero, A_pseudo, l_B+1)
     
    !deallocate
    if (allocated(s)) deallocate(s) 
    if (allocated(u)) deallocate(u)
    if (allocated(vt)) deallocate(vt)
    if (allocated(ss)) deallocate(ss)
    if (allocated(u_H)) deallocate(u_H)
    if (allocated(Sigma_u_H)) deallocate(Sigma_u_H)
   
    !
    allocate(tempMatrix1(l_B+1, n1*n2*n3))
    call zgemm('N', 'N', l_B+1, n1*n2*n3, data_num_bm, one, A_pseudo, l_B+1, va_rm2D, data_num_bm, zero, tempMatrix1, l_B+1)
    
    if (allocated(A_pseudo)) deallocate(A_pseudo)
    
    !
    if (.not. allocated(va_rt2D)) allocate(va_rt2D(data_num_ft, n1*n2*n3))
    if (.not. allocated(temp_bose_ulx_fermi)) allocate(temp_bose_ulx_fermi(data_num_ft,l_B+1))
    temp_bose_ulx_fermi=bose_ulx_fermi
    !va_rt2D(:,:)=matmul(temp_bose_ulx_fermi, tempMatrix1)
    !transform from l to tau (from boson to fermion). *This va_rt2D following is used in convolution
    call zgemm('N', 'N', data_num_ft, n1*n2*n3, l_B+1, one, temp_bose_ulx_fermi, data_num_ft, tempMatrix1, l_B+1, zero, va_rt2D, data_num_ft)
    deallocate(tempMatrix1) !tempMatrix1 is free
    if (allocated(temp_bose_ulx_fermi)) deallocate(temp_bose_ulx_fermi)
    if (allocated(va_rm2D)) deallocate(va_rm2D)
    if (allocated(temp_bose_uln)) deallocate(temp_bose_uln)
    
    !vbg = fft(vb)
    call fft_k(vb, n1, n2, n3, n4F)
    
    if (.not. allocated(vb_rm2D)) allocate(vb_rm2D(n4F,n1*n2*n3))
    do n=1, n4F
       do i=1,n1
          do j=1,n2
             do k=1,n3
                vb_rm2D(n,(i-1)*n2*n3+(j-1)*n3+k)=vb(i,j,k,n)
             end do
          end do
       end do
    end do

    if (.not. allocated(temp_fermi_uln)) allocate(temp_fermi_uln(data_num_fm,l_F+1))
    temp_fermi_uln=fermi_uln
    
    comp=min(data_num_fm, l_F+1)
    if (.not. allocated(work)) allocate(work(1))
    if (.not. allocated(rwork)) allocate(rwork(comp))
    if (.not. allocated(s)) allocate(s(comp))
    if (.not. allocated(u)) allocate(u(data_num_fm, data_num_fm))
    if (.not. allocated(vt)) allocate(vt(l_F+1, l_F+1))

    !workspace 
    lwork = -1
    call zgesvd('S','S',data_num_fm,l_F+1,temp_fermi_uln,data_num_fm,s,u, data_num_fm,vt, l_F+1, work, lwork, rwork, info)

    !SVD by ?gesvd
    lwork=int(work(1))
    deallocate(work)
    allocate(work(lwork))
    call zgesvd('S','S',data_num_fm,l_F+1,temp_fermi_uln,data_num_fm,s,u, data_num_fm,vt, l_F+1, work, lwork, rwork, info)
    deallocate(work)

    !inverse of singular values
    if (.not. allocated(ss)) allocate(ss(comp)) 
    do i=1,comp
      if (s(i)>=1.0d-9*maxval(s(:))) then 
       ss(i)=1.0d0/s(i)
      else
       ss(i)=s(i)
      end if
    end do

    !calculate pseudo inverse, step 1
    if (.not. allocated(u_H)) allocate(u_H(data_num_fm,data_num_fm))
    if (.not. allocated(Sigma_u_H)) allocate(Sigma_u_H(l_F+1,data_num_fm))
    u_H=transpose(conjg(u))
    do i=1,l_F+1
      Sigma_u_H(i,:)=ss(i)*u_H(i,:)
    end do
    
    !calculate pseudo inverse, step 2
    if (.not. allocated(A_pseudo)) allocate(A_pseudo(l_F+1,data_num_fm))
    call zgemm('C', 'N', l_F+1, data_num_fm, l_F+1, one, vt, l_F+1, Sigma_u_H, l_F+1, zero, A_pseudo, l_F+1)

    !deallocate
    if (allocated(s)) deallocate(s) 
    if (allocated(u)) deallocate(u)
    if (allocated(vt)) deallocate(vt)
    if (allocated(ss)) deallocate(ss)
    if (allocated(u_H)) deallocate(u_H)
    if (allocated(Sigma_u_H)) deallocate(Sigma_u_H)

    !
    allocate(tempMatrix2(l_F+1, n1*n2*n3))
    call zgemm('N', 'N', l_F+1, n1*n2*n3, data_num_fm, one, A_pseudo, l_F+1, vb_rm2D, data_num_fm, zero, tempMatrix2, l_F+1)
    
    if (allocated(A_pseudo)) deallocate(A_pseudo)
    

    if (.not. allocated(vb_rt2D)) allocate(vb_rt2D(data_num_ft, n1*n2*n3))
    if (.not. allocated(temp_fermi_ulx_C)) allocate(temp_fermi_ulx_C(data_num_ft,l_F+1))
    temp_fermi_ulx_C=fermi_ulx
    !vb_rt2D=matmul(temp_fermi_ulx, tempMatrix2)
    !transform from l to tau. *This vb_rt2D following is used in convolution
    call zgemm('N', 'N', data_num_ft, n1*n2*n3, l_F+1, one, temp_fermi_ulx_C, data_num_ft, tempMatrix2, l_F+1, zero, vb_rt2D, data_num_ft)
    deallocate(tempMatrix2) !tempMatrix2 is free
    if (allocated(temp_bose_ulx_fermi)) deallocate(temp_bose_ulx_fermi)
    if (allocated(temp_fermi_ulx_C)) deallocate(temp_fermi_ulx_C)
    if (allocated(vb_rm2D)) deallocate(vb_rm2D)
    if (allocated(temp_fermi_uln)) deallocate(temp_fermi_uln)

    !vcg(1:n1, 1:n2, 1:n3, 1:n4) = vag(1:n1, 1:n2, 1:n3, 1:n4) * vbg(1:n1, 1:n2, 1:n3, 1:n4)   
    if (.not. allocated(vc_rt2D)) allocate(vc_rt2D(data_num_ft, n1*n2*n3))
    
    vc_rt2D(1:data_num_ft, 1:n1*n2*n3) = va_rt2D(1:data_num_ft, 1:n1*n2*n3) * vb_rt2D(1:data_num_ft, 1:n1*n2*n3)
    
    if (allocated(vb_rt2D)) deallocate(vb_rt2D)
    if (allocated(va_rt2D)) deallocate(va_rt2D)


    
    !vc = fft_inv(vcg)
    
    if (.not. allocated(temp_fermi_ulx_C)) allocate(temp_fermi_ulx_C(data_num_ft,l_F+1))
    temp_fermi_ulx_C=fermi_ulx
        
    !allocate
    comp=min(data_num_ft, l_F+1)
    if (.not. allocated(work)) allocate(work(1))
    if (.not. allocated(rwork)) allocate(rwork(comp))
    if (.not. allocated(s)) allocate(s(comp))
    if (.not. allocated(u)) allocate(u(data_num_ft, data_num_ft))
    if (.not. allocated(vt)) allocate(vt(l_F+1, l_F+1))

    !workspace 
    lwork = -1
    call zgesvd('S','S',data_num_ft,l_F+1,temp_fermi_ulx_C,data_num_ft,s,u, data_num_ft,vt, l_F+1, work, lwork, rwork, info)

    !SVD by ?gesvd
    lwork=int(work(1))
    deallocate(work)
    allocate(work(lwork))
    call zgesvd('S','S',data_num_ft,l_F+1,temp_fermi_ulx_C,data_num_ft,s,u, data_num_ft,vt, l_F+1, work, lwork, rwork, info)
    deallocate(work)

    !inverse of singular values
    if (.not. allocated(ss)) allocate(ss(comp)) 
    do i=1,comp
      if (s(i)>=1.0d-9*maxval(s(:))) then 
       ss(i)=1.0d0/s(i)
      else
       ss(i)=s(i)
      end if
    end do

    !calculate pseudo inverse, step 1
    if (.not. allocated(u_H)) allocate(u_H(data_num_ft,data_num_ft))
    if (.not. allocated(Sigma_u_H)) allocate(Sigma_u_H(l_F+1,data_num_ft))
    u_H=transpose(conjg(u))
    do i=1,l_F+1
      Sigma_u_H(i,:)=ss(i)*u_H(i,:)
    enddo

    !calculate pseudo inverse, step 2
    if (.not. allocated(A_pseudo)) allocate(A_pseudo(l_F+1,data_num_ft))
    call zgemm('C', 'N', l_F+1, data_num_ft, l_F+1, one, vt, l_F+1, Sigma_u_H, l_F+1, zero, A_pseudo, l_F+1)
     
    !deallocate
    if (allocated(s)) deallocate(s) 
    if (allocated(u)) deallocate(u)
    if (allocated(vt)) deallocate(vt)
    if (allocated(ss)) deallocate(ss)
    if (allocated(u_H)) deallocate(u_H)
    if (allocated(Sigma_u_H)) deallocate(Sigma_u_H)

    !
    allocate(tempMatrix2(l_F+1, n1*n2*n3))
    call zgemm('N', 'N', l_F+1, n1*n2*n3, data_num_ft, one, A_pseudo, l_F+1, vc_rt2D, data_num_ft, zero, tempMatrix2, l_F+1)
    
    if (allocated(A_pseudo)) deallocate(A_pseudo)
    if (.not. allocated(vc_rm2D)) allocate(vc_rm2D(data_num_fm, n1*n2*n3))
    if (.not. allocated(temp_fermi_uln)) allocate(temp_fermi_uln(data_num_fm,l_F+1))
    temp_fermi_uln=fermi_uln
    !vc_rm2D=matmul(temp_fermi_uln, tempMatrix3)
    !transform from l to omega
    call zgemm('N', 'N', data_num_fm, n1*n2*n3, l_F+1, one, temp_fermi_uln, data_num_fm, tempMatrix2, l_F+1, zero, vc_rm2D, data_num_fm)
    
    if (allocated(temp_fermi_uln)) deallocate(temp_fermi_uln)
    if (allocated(temp_fermi_ulx)) deallocate(temp_fermi_ulx)
    if (allocated(temp_fermi_ulx_C)) deallocate(temp_fermi_ulx_C)
    if (allocated(vc_rt2D)) deallocate(vc_rt2D)
    deallocate(tempMatrix2)

    do n=1, n4F
       do i=1,n1
          do j=1,n2
             do k=1,n3
                vc(i,j,k,n)=vc_rm2D(n,(i-1)*n2*n3+(j-1)*n3+k) !vc is actually vc_rm now
             end do
          end do
       end do
    end do

    if (allocated(vc_rm2D)) deallocate(vc_rm2D)
   
    call ifft_k(vc, n1, n2, n3, n4F)    
    va=va_in
    vb=vb_in
    deallocate(va_in)
    deallocate(vb_in)
  end subroutine multiply_fft_bff

  subroutine multiply_fft_tau(vb, vc_rt2D, n1,n2,n3,n4F, n4Ft)
    integer, intent(in):: n1, n2, n3, n4F, n4Ft
    integer:: i,j,k,n
    integer:: info, lwork
    complex(8):: vb(n1,n2,n3,n4F), vc_rt2D(1,n1*n2*n3)
    complex(8), allocatable:: va_in(:,:,:,:), vb_in(:,:,:,:)
    complex(8), allocatable:: va_rm2D(:,:), vb_rm2D(:,:), vc_rm2D(:,:)
    complex(8), allocatable:: va_rt2D(:,:), vb_rt2D(:,:)
    complex(8), allocatable:: temp_bose_uln(:,:), temp_fermi_uln(:,:), temp_bose_ulx_fermi(:,:), temp_fermi_ulx_C(:,:)
    complex(8), allocatable:: tempMatrix1(:,:), tempMatrix2(:,:)
    real(8), allocatable:: temp_fermi_ulx(:,:)
    complex(8), allocatable:: u(:,:), vt(:,:), u_H(:,:), A_pseudo(:,:), Sigma_u_H(:,:)
    real(8), allocatable:: s(:)
    real(8), allocatable:: ss(:)
    logical, save:: initialized = .false.
    integer:: nrank, rank(3), comp
    complex(8) :: zero = (0.0d0, 0.0d0)
    complex(8) :: one = (1.0d0, 0.0d0)
    complex(8), allocatable:: work(:), rwork(:)

    nrank = 3
    rank(1:3) = (/ n1, n2, n3 /)
    allocate(vb_in(n1,n2,n3,n4F))
    vb_in=vb

    call fft_k(vb, n1, n2, n3, n4F)
    
    if (.not. allocated(vb_rm2D)) allocate(vb_rm2D(n4F,n1*n2*n3))
    do n=1, n4F
       do i=1,n1
          do j=1,n2
             do k=1,n3
                vb_rm2D(n,(i-1)*n2*n3+(j-1)*n3+k)=vb(i,j,k,n)
             end do
          end do
       end do
    end do 
    if (.not. allocated(temp_fermi_uln)) allocate(temp_fermi_uln(data_num_fm,l_F+1))
    temp_fermi_uln=fermi_uln

    !allocate
    comp=min(data_num_fm, l_F+1)
    if (.not. allocated(work)) allocate(work(1))
    if (.not. allocated(rwork)) allocate(rwork(comp))
    if (.not. allocated(s)) allocate(s(comp))
    if (.not. allocated(u)) allocate(u(data_num_fm, data_num_fm))
    if (.not. allocated(vt)) allocate(vt(l_F+1, l_F+1))
    !call LA_GELS(temp_fermi_uln,vb_rm2D) !transform from omega to l

    !workspace 
    lwork = -1
    call zgesvd('S','S',data_num_fm,l_F+1,temp_fermi_uln,data_num_fm,s,u, data_num_fm,vt, l_F+1, work, lwork, rwork, info)
    
    !SVD by ?gesvd
    lwork=int(work(1))
    deallocate(work)
    allocate(work(lwork))
    call zgesvd('S','S',data_num_fm,l_F+1,temp_fermi_uln,data_num_fm,s,u, data_num_fm,vt, l_F+1, work, lwork, rwork, info)
    deallocate(work)

    !inverse of singular values
    if (.not. allocated(ss)) allocate(ss(comp)) 
    do i=1,comp
      if (s(i)>=1.0d-9*maxval(s(:))) then 
       ss(i)=1.0d0/s(i)
      else
       ss(i)=s(i)
      end if
    end do

    !calculate pseudo inverse, step 1
    if (.not. allocated(u_H)) allocate(u_H(data_num_fm,data_num_fm))
    if (.not. allocated(Sigma_u_H)) allocate(Sigma_u_H(l_F+1,data_num_fm))
    u_H=transpose(conjg(u))
    do i=1,l_F+1
      Sigma_u_H(i,:)=ss(i)*u_H(i,:)
    enddo
    
    !calculate pseudo inverse, step 2
    if (.not. allocated(A_pseudo)) allocate(A_pseudo(l_F+1,data_num_fm))
    call zgemm('C', 'N', l_F+1, data_num_fm, l_F+1, one, vt, l_F+1, Sigma_u_H, l_F+1, zero, A_pseudo, l_F+1)
     
    !deallocate
    if (allocated(s)) deallocate(s) 
    if (allocated(u)) deallocate(u)
    if (allocated(vt)) deallocate(vt)
    if (allocated(ss)) deallocate(ss)
    if (allocated(u_H)) deallocate(u_H)
    if (allocated(Sigma_u_H)) deallocate(Sigma_u_H)
   
    !
    allocate(tempMatrix2(l_F+1, n1*n2*n3))
    call zgemm('N', 'N', l_F+1, n1*n2*n3, data_num_fm, one, A_pseudo, l_F+1, vb_rm2D, data_num_fm, zero, tempMatrix2, l_F+1)
    
    if (allocated(A_pseudo)) deallocate(A_pseudo)
    !
    if (.not. allocated(vb_rt2D)) allocate(vb_rt2D(1, n1*n2*n3))
    if (.not. allocated(temp_fermi_ulx_C)) allocate(temp_fermi_ulx_C(1,l_F+1))
    temp_fermi_ulx_C=fermi_ulx_0
    !vb_rt2D=matmul(temp_fermi_ulx, tempMatrix2)
    !transform from l to tau. *This vb_rt2D following is used in convolution
    call zgemm('N', 'N', 1, n1*n2*n3, l_F+1, one, temp_fermi_ulx_C, 1, tempMatrix2, l_F+1, zero, vb_rt2D, 1)
    deallocate(tempMatrix2) !tempMatrix2 is free
    if (allocated(temp_bose_ulx_fermi)) deallocate(temp_bose_ulx_fermi)
    if (allocated(temp_fermi_ulx_C)) deallocate(temp_fermi_ulx_C)
    if (allocated(vb_rm2D)) deallocate(vb_rm2D)
    if (allocated(temp_fermi_uln)) deallocate(temp_fermi_uln)

    !vcg(1:n1, 1:n2, 1:n3, 1:n4) = vag(1:n1, 1:n2, 1:n3, 1:n4) * vbg(1:n1, 1:n2, 1:n3, 1:n4)  
    
    vc_rt2D(1, 1:n1*n2*n3) = vb_rt2D(1, 1:n1*n2*n3)
    
    if (allocated(vb_rt2D)) deallocate(vb_rt2D)
    if (allocated(va_rt2D)) deallocate(va_rt2D)

    !vc = fft_inv(vcg)
    
    vb=vb_in
    deallocate(vb_in)

  end subroutine multiply_fft_tau
  
  subroutine multiply_fft_W(va, vb, vc, n1, n2, n3, n4F, n4Ft)
    integer , intent(in):: n1, n2, n3, n4F, n4Ft
    integer:: i,j,k,n
    integer:: info, lwork
    complex(8):: va(n1,n2,n3), vb(n1,n2,n3,n4F), vc(n1,n2,n3,1)
    complex(8), allocatable:: va_in(:,:,:), vb_in(:,:,:,:), temp(:,:,:,:)
    complex(8), allocatable:: va_rm2D(:,:), vb_rm2D(:,:), vc_rm2D(:,:)
    complex(8), allocatable:: temp_bose_uln(:,:), temp_fermi_uln(:,:),temp_bose_ulx_fermi(:,:), temp_fermi_ulx_C(:,:)
    complex(8), allocatable::tempMatrix1(:,:), tempMatrix2(:,:)
    complex(8), allocatable:: u(:,:), vt(:,:), u_H(:,:), A_pseudo(:,:), Sigma_u_H(:,:)
    real(8), allocatable:: temp_fermi_ulx(:,:)
    real(8), allocatable:: s(:)
    real(8), allocatable:: ss(:)
    integer:: nrank, rank(3), comp
    complex(8) :: zero =(0.0d0, 0.0d0)
    complex(8) :: one = (1.0d0, 0.0d0)   
    complex(8), allocatable:: work(:), rwork(:)

    nrank = 3
    rank(1:3) = (/ n1, n2, n3/)
    allocate(va_in(n1,n2,n3))
    allocate(vb_in(n1,n2,n3,n4F))
    va_in = va
    vb_in = vb 
  
    call fft_k_3d(va,n1,n2,n3)
    call fft_k(vb,n1,n2,n3,n4F)

    allocate(temp(n1,n2,n3,n4F))
    do n=1,n4F
     temp(1:n1,1:n2,1:n3,n)=va(1:n1,1:n2,1:n3)*vb(1:n1,1:n2,1:n3,n)
    end do
  
    vb=temp
    deallocate(temp)
  
    call ifft_k(vb,n1,n2,n3,n4F)

    if(.not. allocated(vb_rm2D)) allocate(vb_rm2D(n4F, n1*n2*n3))
    do n=1, n4F
     do i=1, n1
        do j=1, n2
           do k=1, n3
              vb_rm2D(n,(i-1)*n2*n3+(j-1)*n3+k)=vb(i,j,k,n)
           end do
        end do
     end do
    end do

    if (.not. allocated(temp_fermi_uln)) allocate(temp_fermi_uln(data_num_fm,l_F+1))
    temp_fermi_uln=fermi_uln

    comp=min(data_num_fm, l_F+1)
    if (.not. allocated(work)) allocate(work(1))
    if (.not. allocated(rwork)) allocate(rwork(comp))
    if (.not. allocated(s)) allocate(s(comp))
    if (.not. allocated(u)) allocate(u(data_num_fm, data_num_fm))
    if (.not. allocated(vt)) allocate(vt(l_F+1, l_F+1))

    !workspace 
    lwork = -1
    call zgesvd('S','S',data_num_fm,l_F+1,temp_fermi_uln,data_num_fm,s,u, data_num_fm,vt, l_F+1, work, lwork, rwork, info)
    
    !SVD by ?gesvd
    lwork=int(work(1))
    deallocate(work)
    allocate(work(lwork))
    call zgesvd('S','S',data_num_fm,l_F+1,temp_fermi_uln,data_num_fm,s,u, data_num_fm,vt, l_F+1, work, lwork, rwork, info)
    deallocate(work)

    !inverse of singular values
    if (.not. allocated(ss)) allocate(ss(comp)) 
    do i=1,comp
      if (s(i)>=1.0d-9*maxval(s(:))) then 
       ss(i)=1.0d0/s(i)
      else
       ss(i)=s(i)
      end if
    end do

    !calculate pseudo inverse, step 1
    if (.not. allocated(u_H)) allocate(u_H(data_num_fm,data_num_fm))
    if (.not. allocated(Sigma_u_H)) allocate(Sigma_u_H(l_F+1,data_num_fm))
    u_H=transpose(conjg(u))
    do i=1,l_F+1
      Sigma_u_H(i,:)=ss(i)*u_H(i,:)
    enddo
    
    !calculate pseudo inverse, step 2
    if (.not. allocated(A_pseudo)) allocate(A_pseudo(l_F+1,data_num_fm))
    call zgemm('C', 'N', l_F+1, data_num_fm, l_F+1, one, vt, l_F+1, Sigma_u_H, l_F+1, zero, A_pseudo, l_F+1)
     
    !deallocate
    if (allocated(s)) deallocate(s) 
    if (allocated(u)) deallocate(u)
    if (allocated(vt)) deallocate(vt)
    if (allocated(ss)) deallocate(ss)
    if (allocated(u_H)) deallocate(u_H)
    if (allocated(Sigma_u_H)) deallocate(Sigma_u_H)
   
    !
    allocate(tempMatrix2(l_F+1, n1*n2*n3))
    call zgemm('N', 'N', l_F+1, n1*n2*n3, data_num_fm, one, A_pseudo, l_F+1, vb_rm2D, data_num_fm, zero, tempMatrix2, l_F+1)
    
    if (allocated(A_pseudo)) deallocate(A_pseudo)


  if (.not. allocated(vc_rm2D)) allocate(vc_rm2D(1, n1*n2*n3))
  if (.not. allocated(temp_fermi_ulx_C)) allocate(temp_fermi_ulx_C(1,l_F+1))
  temp_fermi_ulx_C=fermi_ulx_0
  call zgemm('N', 'N', 1, n1*n2*n3, l_F+1, one, temp_fermi_ulx_C, 1, tempMatrix2, l_F+1, zero, vc_rm2D, 1)
  deallocate(tempMatrix2)
    
  if (allocated(temp_bose_ulx_fermi)) deallocate(temp_bose_ulx_fermi)
  if (allocated(temp_fermi_ulx_C)) deallocate(temp_fermi_ulx_C)
  if (allocated(vb_rm2D)) deallocate(vb_rm2D)
  if (allocated(temp_fermi_uln)) deallocate(temp_fermi_uln)
  
  do i=1,n1
     do j=1,n2
        do k=1,n3
           vc(i,j,k,1)=vc_rm2D(1,(i-1)*n2*n3+(j-1)*n3+k)
        end do
     end do
  end do
  if (allocated(vc_rm2D)) deallocate(vc_rm2D)
  
  va=va_in
  vb=vb_in
  deallocate(va_in)
  deallocate(vb_in)
 
  end subroutine


  subroutine fft_k(va,n1,n2,n3,n4)
    integer, intent(in):: n1, n2, n3, n4
    integer:: n
    complex(8):: va(n1,n2,n3,n4)
    complex(8), allocatable:: va_temp(:,:,:)
    include "fftw3.f"
    integer(8):: plan
    integer:: nrank, rank(3)

    nrank = 3
    rank(1:3) = (/ n1, n2, n3 /)
    if (.not. allocated(va_temp)) allocate(va_temp(n1,n2,n3))
    do n=1,n4
       va_temp=0
       call dfftw_plan_dft(plan, nrank, rank, va(:,:,:,n), va_temp, FFTW_FORWARD, FFTW_ESTIMATE)
       call dfftw_execute(plan)
       call dfftw_destroy_plan(plan)
       va(:,:,:,n) = va_temp(:,:,:)
    end do
  end subroutine fft_k
   
  subroutine fft_k_3d(va,n1,n2,n3)
    integer, intent(in):: n1,n2,n3
    integer:: n
    complex(8):: va(n1,n2,n3)
    complex(8), allocatable:: va_temp(:,:,:)
    include "fftw3.f"
    integer(8):: plan
    integer:: nrank, rank(3)
 
    nrank = 3
    rank(1:3) = (/ n1, n2, n3/)
    if (.not. allocated(va_temp)) allocate(va_temp(n1,n2,n3))
    call dfftw_plan_dft(plan, nrank, rank, va(:,:,:), va_temp(:,:,:),FFTW_BACKWARD, FFTW_ESTIMATE)
    call dfftw_execute(plan)
    call dfftw_destroy_plan(plan)
    va(:,:,:)=va_temp(:,:,:)
    deallocate(va_temp)
    end subroutine

     

  subroutine ifft_k(va,n1,n2,n3,n4)
    integer, intent(in):: n1, n2, n3, n4
    integer:: n
    complex(8):: va(n1,n2,n3,n4)
    complex(8), allocatable:: va_temp(:,:,:)
    include "fftw3.f"
    integer(8):: plan
    integer:: nrank, rank(3)
    
    nrank = 3
    rank(1:3) = (/ n1, n2, n3 /)
    if (.not. allocated(va_temp)) allocate(va_temp(n1,n2,n3))
    do n=1,n4
       va_temp=0
       call dfftw_plan_dft(plan, nrank, rank, va(:,:,:,n), va_temp, FFTW_BACKWARD, FFTW_ESTIMATE)
       call dfftw_execute(plan)
       call dfftw_destroy_plan(plan)
       va(:,:,:,n) = va_temp(:,:,:)/(n1*n2*n3)
    end do
  end subroutine ifft_k
  
  subroutine multiply_fft(va, vb, vc, n1, n2, n3, n4)
    integer, intent(in):: n1, n2, n3, n4
    complex(8):: va(n1,n2,n3,n4), vb(n1,n2,n3,n4), vc(n1,n2,n3,n4)
    complex(8), allocatable:: vag(:,:,:,:)
    include "fftw3.f"
    integer(8), save:: plan1, plan2, plan3
    logical, save:: initialized = .false.
    integer:: nrank, rank(4)
    allocate (vag(n1,n2,n3,n4))
    nrank = 4
    rank(1:4) = (/ n1, n2, n3, n4 /)
    !vag = fft(va)
    vag = 0.d0
    if(.not.initialized) call dfftw_plan_dft(plan1, nrank, rank, va, vag, FFTW_FORWARD, FFTW_ESTIMATE)
    call dfftw_execute(plan1)
    !call dfftw_destroy_plan(plan1)
    !vbg = fft(vb)
    !vbg = fft(vb)
    !vbg = 0.d0
    !call dfftw_plan_dft(plan2, nrank, rank, vb, vbg, FFTW_FORWARD, FFTW_ESTIMATE)
    vc = 0.d0
    if(.not.initialized) call dfftw_plan_dft(plan2, nrank, rank, vb, vc, FFTW_FORWARD, FFTW_ESTIMATE)
    call dfftw_execute(plan2)
    !call dfftw_destroy_plan(plan2)
    !vag(1:n1, 1:n2, 1:n3, 1:n4) = vag(1:n1, 1:n2, 1:n3, 1:n4) * vbg(1:n1, 1:n2, 1:n3, 1:n4)
    vag(1:n1, 1:n2, 1:n3, 1:n4) = vag(1:n1, 1:n2, 1:n3, 1:n4) * vc(1:n1, 1:n2, 1:n3, 1:n4)
    !vc = fft_inv(vcg)
    vc = 0.d0
    if(.not.initialized) call dfftw_plan_dft(plan3, nrank, rank, vag, vc, FFTW_BACKWARD, FFTW_ESTIMATE)
    call dfftw_execute(plan3)
    !call dfftw_destroy_plan(plan3)
    vc(1:n1, 1:n2, 1:n3, 1:n4) = vc(1:n1, 1:n2, 1:n3, 1:n4) / (n1 * n2 * n3 * n4)
    deallocate(vag)
  end subroutine multiply_fft
end module fft