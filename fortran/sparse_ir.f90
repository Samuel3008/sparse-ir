module sparse_ir
    implicit none

    type DecomposedMatrix
        double precision, allocatable :: inv_s(:)
        complex(kind(0d0)), allocatable :: a(:, :)
        complex(kind(0d0)), allocatable :: ut(:, :), v(:, :)
    end type

    type IR
        integer::size
        complex(kind(0d0)), allocatable :: u(:, :)
        complex(kind(0d0)), allocatable :: uhat(:, :)
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

        ! Count number of singular values
        ns = 0
        do i = 1, mn
            if (s(i)/s(1) < eps) then
                exit
            end if
            ns = ns + 1
        end do

        allocate(dmat%a(m, n))
        allocate(dmat%ut(ns, m))
        allocate(dmat%v(n, ns))

        dmat%a(1:m, 1:n) = a(1:m, 1:n)
        dmat%inv_s(1:ns) = 1/s(1:ns)
        dmat%ut(1:ns, 1:m) = conjg(transpose(u(1:m, 1:ns)))
        dmat%v(1:n, 1:ns) = conjg(transpose(vt(1:ns, 1:n)))

        deallocate(work, a_copy, s, u, vt, work, iwork)
    end function


    function read(unit) result(obj)
        !character (len=*), intent (in) :: filename
        integer, intent (in) :: unit

        type(IR)::obj
        allocate(obj%u(obj%size, 10))
        allocate(obj%uhat(obj%size, 10))

    end function

end module
