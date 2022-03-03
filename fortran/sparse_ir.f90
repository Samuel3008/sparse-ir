module sparse_ir
    implicit none

    ! Matrix decomposed in SVD for fitting
    type DecomposedMatrix
        complex(kind(0d0)), allocatable :: a(:, :) ! Original matrix
        double precision, allocatable :: inv_s(:) ! Inverse of singular values
        complex(kind(0d0)), allocatable :: ut(:, :), v(:, :)
    end type

    ! Sampling points, basis functions
    type IR
        integer :: size, ntau, nfreq_f, nfreq_b
        double precision :: lambda, eps
        double precision, allocatable :: s(:), tau(:)
        integer, allocatable :: freq_f(:), freq_b(:)
        complex(kind(0d0)), allocatable :: u(:, :)
        complex(kind(0d0)), allocatable :: uhat_f(:, :), uhat_b(:, :)
        type(DecomposedMatrix) :: u_fit
        type(DecomposedMatrix) :: uhat_fit_f, uhat_fit_b
    end type

    contains

    ! SVD of matrix a. Singular values smaller than esp * the largest one are dropped.
    function decompose(a, eps) result(dmat)
        complex(kind(0d0)), intent(in) :: a(:, :)
        double precision, intent(in) :: eps

        integer :: i, info, lda, ldu, ldvt, lwork, m, n, mn, mx, ns
        complex(kind(0d0)), allocatable :: a_copy(:, :), u(:, :), &
            vt(:, :), work(:)
        double precision, allocatable :: rwork(:), s(:)
        integer, allocatable :: iwork(:)
        type(DecomposedMatrix)::dmat

        m = size(a, 1)
        n = size(a, 2)
        mx = max(m, n)
        mn = min(m, n)
        lda = m
        ldu = m
        ldvt = n
        lwork = mn*mn + 3*mn

        allocate(work(lwork), a_copy(m,n), s(m), u(ldu,m), vt(ldvt,n), rwork((5*mn+7)*mn), iwork(8*mn))

        a_copy(1:m, 1:n) = a(1:m, 1:n)
        call zgesdd('S', m, n, a_copy, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info)

        if (info /= 0) then
            write (*, *) 'Failure in ZGESDD. INFO =', info
            stop
        end if

        ! Number of relevant singular values s(i)/s(1) >= eps
        ns = 0
        do i = 1, mn
            if (s(i)/s(1) < eps) then
                exit
            end if
            ns = ns + 1
        end do

        allocate(dmat%a(m, n))
        allocate(dmat%inv_s(ns))
        allocate(dmat%ut(ns, m))
        allocate(dmat%v(n, ns))

        dmat%a = a
        dmat%inv_s(1:ns) = 1/s(1:ns)
        dmat%ut(1:ns, 1:m) = conjg(transpose(u(1:m, 1:ns)))
        dmat%v(1:n, 1:ns) = conjg(transpose(vt(1:ns, 1:n)))

        deallocate(work, a_copy, s, u, vt, rwork, iwork)
    end function

    ! Read sampling points, basis functions
    function read(unit) result(obj)
        integer, intent (in) :: unit
        integer :: version
        character(len=100) :: tmp_str
        type(IR) :: obj

        read(unit,*) tmp_str, version
        if (version == 1) then
            call read_v1(unit, obj)
        else
            write(*, *) "Invalid version number", version
            stop
        end if
    end

    ! Read sampling points, basis functions (version 1)
    subroutine read_v1(unit, obj)
        integer, intent (in) :: unit
        type(IR), intent (inout) :: obj

        character(len=100) :: tmp_str
        integer :: i, l, t, n
        double precision, parameter :: rtol = 1e-20

        read(unit,*) tmp_str, obj%lambda
        read(unit,*) tmp_str, obj%eps

        ! Singular values
        read(unit,*)
        read(unit,*) obj%size
        allocate(obj%s(obj%size))
        do i=1, obj%size
            read(unit, *) obj%s(i)
        end do

        ! Sampling times
        read(unit,*)
        read(unit,*) obj%ntau
        allocate(obj%tau(obj%ntau))
        do i=1, obj%ntau
            read(unit, *) obj%tau(i)
        end do

        ! Basis functions on sampling times
        read(unit,*)
        allocate(obj%u(obj%ntau, obj%size))
        do l = 1, obj%size
            do t = 1, obj%ntau
                read(unit, *) obj%u(t, l)
            end do
        end do
        obj%u_fit = decompose(obj%u, rtol)

        ! Sampling frequencies (F)
        read(unit,*)
        read(unit,*) obj%nfreq_f
        allocate(obj%freq_f(obj%nfreq_f))
        do i=1, obj%nfreq_f
            read(unit, *) obj%freq_f(i)
        end do

        read(unit,*)
        allocate(obj%uhat_f(obj%nfreq_f, obj%size))
        do l = 1, obj%size
            do n = 1, obj%nfreq_f
                read(unit, *) obj%uhat_f(n, l)
            end do
        end do
        obj%uhat_fit_f = decompose(obj%uhat_f, rtol)

        ! Sampling frequencies (B)
        read(unit,*)
        read(unit,*) obj%nfreq_b
        allocate(obj%freq_b(obj%nfreq_b))
        do i=1, obj%nfreq_b
            read(unit, *) obj%freq_b(i)
        end do

        read(unit,*)
        allocate(obj%uhat_b(obj%nfreq_b, obj%size))
        do l = 1, obj%size
            do n = 1, obj%nfreq_b
                read(unit, *) obj%uhat_b(n, l)
            end do
        end do
        obj%uhat_fit_b = decompose(obj%uhat_f, rtol)

    end

end module
