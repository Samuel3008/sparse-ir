module sparse_ir
    implicit none

    ! Matrix decomposed in SVD for fitting
    type DecomposedMatrix
        complex(kind(0d0)), allocatable :: a(:, :) ! Original matrix
        double precision, allocatable :: inv_s(:) ! Inverse of singular values
        complex(kind(0d0)), allocatable :: ut(:, :), v(:, :)
        integer :: m, n, ns
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
        dmat%m = size(a, 1)
        dmat%n = size(a, 2)
        dmat%ns = ns

        deallocate(work, a_copy, s, u, vt, rwork, iwork)
    end function

    subroutine fit_matsubara_f(obj, arr, res)
        type(IR), intent(in) :: obj
        complex(kind(0d0)), intent (in) :: arr(:, :)
        complex(kind(0d0)), intent(out) :: res(:, :)
        call fit_impl(arr, obj%uhat_fit_f, res)
    end

    subroutine fit_matsubara_b(obj, arr, res)
        type(IR), intent(in) :: obj
        complex(kind(0d0)), intent (in) :: arr(:, :)
        complex(kind(0d0)), intent(out) :: res(:, :)
        call fit_impl(arr, obj%uhat_fit_b, res)
    end

    subroutine fit_tau(obj, arr, res)
        type(IR), intent(in) :: obj
        complex(kind(0d0)), intent (in) :: arr(:, :)
        complex(kind(0d0)), intent(out) :: res(:, :)
        call fit_impl(arr, obj%u_fit, res)
    end

    subroutine evaluate_matsubara_f(obj, arr, res)
        type(IR), intent(in) :: obj
        complex(kind(0d0)), intent (in) :: arr(:, :)
        complex(kind(0d0)), intent(out) :: res(:, :)
        res = matmul(arr, obj%uhat_fit_f%a)
    end

    subroutine evaluate_matsubara_b(obj, arr, res)
        type(IR), intent(in) :: obj
        complex(kind(0d0)), intent (in) :: arr(:, :)
        complex(kind(0d0)), intent(out) :: res(:, :)
        res = matmul(arr, obj%uhat_fit_b%a)
    end

    ! Implementation of fit
    subroutine fit_impl(arr, mat, res)
        complex(kind(0d0)), intent (in) :: arr(:, :)
        type(DecomposedMatrix), intent(in) :: mat
        complex(kind(0d0)), intent(out) :: res(:, :)

        complex(kind(0d0)), allocatable :: ut_arr(:, :)

        integer :: nb, m, n, ns, i, j

        ! arr(nb, m)
        ! mat(m, n)
        ! ut_arr(nb, ns)
        ! res(nb, n)
        nb = size(arr, 1)
        m = mat%m
        n = mat%n

        if (size(res, 1) /= nb .or. size(res, 2) /= n) then
            write(*, *) 'Invalid size of output array'
            stop
        end if

        allocate(ut_arr(nb, ns))

        ut_arr = matmul(mat%ut, arr)
        do j = 1, ns
            do i = 1, nb
                ut_arr(i, j) = ut_arr(i, j) * mat%inv_s(j)
            end do
        end do

        res = matmul(mat%v, ut_arr)

        deallocate(ut_arr)
    end


end module
