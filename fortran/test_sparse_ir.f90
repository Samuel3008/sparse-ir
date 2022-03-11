program main
    use sparse_ir
    use sparse_ir_io
    use sparse_ir_preset
    implicit none

    call test_fermion(.true.)
    call test_boson(.true.)
    call test_fermion(.false.)
    call test_boson(.false.)
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
    end subroutine

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
    end subroutine

    ! fermion
    subroutine test_fermion(preset)
        logical, intent(in) :: preset
        type(IR) :: ir_obj
        integer, parameter :: ndigit = 10, nlambda = 4
        double precision, parameter :: lambda = 1.d1 ** nlambda
        double precision, parameter :: wmax = 1.d0, PI=4.D0*DATAN(1.D0)

        double precision, parameter :: beta = lambda/wmax, omega0 = 1.d0/beta
        double precision, parameter :: eps = 1.d-1**ndigit

        complex(kind(0d0)),allocatable :: giv(:,:), gl_ref(:, :), gl_matsu(:, :), gl_tau(:, :), gtau(:, :), &
            gtau_reconst(:, :), giv_reconst(:, :)
        integer n, t, l

        if (preset) then
            ir_obj = mk_ir_preset(nlambda, ndigit, beta)
        else
            open(99, file='ir_nlambda4_ndigit10.dat', status='old')
            ir_obj = read_ir(99, beta)
            close(99)
        end if

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
        allocate(gtau_reconst(1, ir_obj%ntau))
        allocate(giv_reconst(1, ir_obj%nfreq_f))

        ! From Matsubara
        do n = 1, ir_obj%nfreq_f
            giv(1, n) = 1.d0/(cmplx(0d0, PI*ir_obj%freq_f(n)/beta, kind(0d0)) - omega0)
        end do
        call fit_matsubara_f(ir_obj, giv, gl_matsu)

        ! From tau
        !   G(τ=0) = - exp(-τ ω0)/(1+exp(-β ω0)),
        do t = 1, ir_obj%ntau
            gtau(1, t) = - exp(-ir_obj%tau(t) * omega0)/(1.d0 + exp(-beta * omega0))
        end do
        call fit_tau(ir_obj, gtau, gl_tau)

        !do l = 1, ir_obj%size
            !write(*,*) real(gl_matsu(1,l)), real(gl_tau(1,l))
        !end do
        if (maxval(abs(gl_matsu - gl_tau)) > 1d2*eps) then
            stop "gl_matsu and gl_tau do not match!"
        end if

        call evaluate_matsubara_f(ir_obj, gl_matsu, giv_reconst)
        if (maxval(abs(giv - giv_reconst)) > 1d2*eps) then
            stop "giv do not match!"
        end if

        call evaluate_tau(ir_obj, gl_tau, gtau_reconst)
        if (maxval(abs(gtau - gtau_reconst)) > 1d2*eps) then
            stop "gtau do not match!"
        end if

        deallocate(giv, gtau, gl_ref, gl_matsu, gl_tau, gtau_reconst, giv_reconst)
    end subroutine


    ! boson
    subroutine test_boson(preset)
        logical, intent(in) :: preset
        type(IR) :: ir_obj
        integer, parameter :: ndigit = 10, nlambda = 4
        double precision, parameter :: lambda = 1.d1 ** nlambda
        double precision, parameter :: wmax = 1.d0, PI=4.D0*DATAN(1.D0)

        double precision, parameter :: beta = lambda/wmax, omega0 = 1.d0/beta
        double precision, parameter :: eps = 1.d-1**ndigit

        complex(kind(0d0)),allocatable :: giv(:,:), gl_ref(:, :), gl_matsu(:, :), gl_tau(:, :), gtau(:, :), &
            gtau_reconst(:, :), giv_reconst(:, :)
        integer n, t

        if (preset) then
            ir_obj = mk_ir_preset(nlambda, ndigit, beta)
        else
            open(99, file='ir_nlambda4_ndigit10.dat', status='old')
            ir_obj = read_ir(99, beta)
            close(99)
        end if

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
        allocate(gtau_reconst(1, ir_obj%ntau))
        allocate(giv_reconst(1, ir_obj%nfreq_b))

        ! From Matsubara
        do n = 1, ir_obj%nfreq_b
            giv(1, n) = 1.d0/(cmplx(0d0, PI*ir_obj%freq_b(n)/beta, kind(0d0)) - omega0)
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

        if (maxval(abs(gl_matsu - gl_tau)) > 1d2*eps) then
            stop "gl_matsu and gl_tau do not match!"
        end if

        call evaluate_matsubara_b(ir_obj, gl_matsu, giv_reconst)
        if (maxval(abs(giv - giv_reconst)) > 1d2*eps) then
            stop "gtau do not match!"
        end if

        call evaluate_tau(ir_obj, gl_tau, gtau_reconst)
        if (maxval(abs(gtau - gtau_reconst)) > 1d2*eps) then
            stop "gtau do not match!"
        end if

        deallocate(giv, gtau, gl_ref, gl_matsu, gl_tau, gtau_reconst, giv_reconst)
    end subroutine

end program