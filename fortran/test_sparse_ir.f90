program main
    use sparse_ir
    use sparse_ir_io
    implicit none

    call test_fermion()
    call test_boson()
    call test_fit()
    call test_fit_rectangular()

    contains

    subroutine test_fit()
        complex(kind(0d0)) :: a(2, 2), y(1, 2), x(1, 2), y_reconst(1, 2)
        type(DecomposedMatrix) :: dm
        a(1, 1) = 2.d0
        a(2, 1) = 0.1d0
        a(1, 2) = 0.d0
        a(2, 2) = 1.d0

        dm = decompose(a, 1d-20)

        y(1, 1) = 0.2d0
        y(1, 2) = 0.1d0

        call fit_impl(y, dm, x)

        y_reconst = transpose(matmul(dm%a, transpose(x)))
        if (maxval(abs(y - y_reconst)) > 1e-12) then
            stop "y and y_reconst do not match!"
        end if
        !write(*, *) y
        !write(*, *) y_reconst
    end

    subroutine test_fit_rectangular()
        integer, parameter :: n=1, m=2
        complex(kind(0d0)) :: a(n, m), y(1, n), x(1, m), y_reconst(1, n)
        type(DecomposedMatrix) :: dm
        a(1, 1) = 2.d0
        a(1, 2) = 1.d0

        dm = decompose(a, 1d-10)

        y(1, 1) = 0.2d0

        call fit_impl(y, dm, x)

        y_reconst = transpose(matmul(dm%a, transpose(x)))
        if (maxval(abs(y - y_reconst)) > 1e-12) then
            stop "y and y_reconst do not match!"
        end if
    end

    ! fermion
    subroutine test_fermion()
        type(IR) :: ir_obj
        double precision, parameter :: lambda = 1d+4
        integer, parameter :: ndigit = 10
        double precision, parameter :: wmax = 1.d0, PI=4.D0*DATAN(1.D0)

        double precision, parameter :: beta = lambda/wmax, omega0 = 1/beta
        double precision, parameter :: eps = 1.d0/10.d0**ndigit

        complex(kind(0d0)),allocatable :: giv(:,:), gl_ref(:, :), gl_matsu(:, :), gl_tau(:, :), gtau(:, :)
        integer n, t

        open(99, file='ir_nlambda4_ndigit10.dat', status='old')
        ir_obj = read_ir(99, beta)
        close(99)

        if (abs(ir_obj%beta - beta) > 1d-10) then
            stop "beta does not match"
        end if
        if (abs(ir_obj%wmax - wmax) > 1d-10) then
            stop "wmax does not match"
        end if

        ! With ω0 = 1/β,
        !   G(iv) = 1/(iv - ω0),
        !   G(τ=0) = - exp(-τ ω0)/(1+exp(-β ω0)),
        allocate(giv(1, ir_obj%nfreq_f))
        allocate(gtau(1, ir_obj%ntau))
        allocate(gl_ref(1, ir_obj%size))
        allocate(gl_matsu(1, ir_obj%size))
        allocate(gl_tau(1, ir_obj%size))

        ! From Matsubara
        do n = 1, ir_obj%nfreq_f
            giv(1, n) = 1/(dcmplx(0d0, PI*ir_obj%freq_f(n)/beta) - omega0)
        end do
        call fit_matsubara_f(ir_obj, giv, gl_matsu)

        ! From tau
        !   G(τ=0) = - exp(-τ ω0)/(1+exp(-β ω0)),
        do t = 1, ir_obj%ntau
            gtau(1, t) = - exp(-ir_obj%tau(t) * omega0)/(1.d0 + exp(-beta * omega0))
        end do
        call fit_tau(ir_obj, gtau, gl_tau)

        if (maxval(abs(gl_matsu - gl_tau)) > 100*eps) then
            stop "gl_matsu and gl_tau do not match!"
        end if

        deallocate(giv, gtau, gl_ref, gl_matsu, gl_tau)
    end


    ! boson
    subroutine test_boson()
        type(IR) :: ir_obj
        double precision, parameter :: lambda = 1d+4
        integer, parameter :: ndigit = 10
        double precision, parameter :: wmax = 1.d0, PI=4.D0*DATAN(1.D0)

        double precision, parameter :: beta = lambda/wmax, omega0 = 1/beta
        double precision, parameter :: eps = 1.d0/10.d0**ndigit

        complex(kind(0d0)),allocatable :: giv(:,:), gl_ref(:, :), gl_matsu(:, :), gl_tau(:, :), gtau(:, :)
        integer n, t

        open(99, file='ir_nlambda4_ndigit10.dat', status='old')
        ir_obj = read_ir(99, beta)
        close(99)

        if (abs(ir_obj%beta - beta) > 1d-10) then
            stop "beta does not match"
        end if
        if (abs(ir_obj%wmax - wmax) > 1d-10) then
            stop "wmax does not match"
        end if

        ! With ω0 = 1/β,
        !   G(iv) = 1/(iv - ω0),
        !   G(τ=0) = - exp(-τ ω0)/(1-exp(-β ω0)),
        allocate(giv(1, ir_obj%nfreq_b))
        allocate(gtau(1, ir_obj%ntau))
        allocate(gl_ref(1, ir_obj%size))
        allocate(gl_matsu(1, ir_obj%size))
        allocate(gl_tau(1, ir_obj%size))

        ! From Matsubara
        do n = 1, ir_obj%nfreq_b
            giv(1, n) = 1/(dcmplx(0d0, PI*ir_obj%freq_b(n)/beta) - omega0)
        end do
        call fit_matsubara_b(ir_obj, giv, gl_matsu)

        ! From tau
        !   G(τ=0) = - exp(-τ ω0)/(1-exp(-β ω0)),
        do t = 1, ir_obj%ntau
            gtau(1, t) = - exp(-ir_obj%tau(t) * omega0)/(1.d0 - exp(-beta * omega0))
        end do
        call fit_tau(ir_obj, gtau, gl_tau)

        !do l = 1, ir_obj%size
            !write(*, *) l, real(gl_tau(1, l)), real(gl_matsu(1, l))
        !end do

        if (maxval(abs(gl_matsu - gl_tau)) > 100*eps) then
            stop "gl_matsu and gl_tau do not match!"
        end if

        deallocate(giv, gtau, gl_ref, gl_matsu, gl_tau)
    end

end program