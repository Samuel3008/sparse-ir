module sparse_ir_io
    use sparse_ir
    implicit none

    contains

    ! Read sampling points, basis functions
    function read_ir(unit) result(obj)
        integer, intent (in) :: unit

        type(IR) :: obj
        integer :: version
        character(len=100) :: tmp_str

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
        double precision :: rtmp, rtmp2
        double precision, parameter :: rtol = 1e-20

        read(unit,*) tmp_str, obj%lambda
        read(unit,*) tmp_str, obj%eps

        ! Singular values
        read(unit,*)
        read(unit,*) obj%size
        !write(*, *) "size", obj%size
        allocate(obj%s(obj%size))
        do i=1, obj%size
            read(unit, *) obj%s(i)
            !write(*, *) i, obj%s(i)
        end do

        ! Sampling times
        read(unit,*)
        read(unit,*) obj%ntau
        !write(*, *) "size", obj%ntau
        allocate(obj%tau(obj%ntau))
        do i=1, obj%ntau
            read(unit, *) obj%tau(i)
            !write(*, *) i, obj%tau(i)
        end do

        ! Basis functions on sampling times
        read(unit,*)
        allocate(obj%u(obj%ntau, obj%size))
        do l = 1, obj%size
            do t = 1, obj%ntau
                read(unit, *) rtmp
                obj%u(t, l) = rtmp
                !write(*, *) l, t, obj%u(t, l)
            end do
        end do
        obj%u_fit = decompose(obj%u, rtol)

        ! Sampling frequencies (F)
        read(unit,*)
        read(unit,*) obj%nfreq_f
        allocate(obj%freq_f(obj%nfreq_f))
        do i=1, obj%nfreq_f
            read(unit, *) obj%freq_f(i)
            !write(*, *) 'freq', i, obj%freq_f(i)
        end do

        read(unit,*)
        allocate(obj%uhat_f(obj%nfreq_f, obj%size))
        do l = 1, obj%size
            do n = 1, obj%nfreq_f
                read(unit, *) rtmp, rtmp2
                obj%uhat_f(n, l) = dcmplx(rtmp, rtmp2)
                !write(*, *) l, n, obj%uhat_f(n, l)
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
                read(unit, *) rtmp, rtmp2
                obj%uhat_b(n, l) = dcmplx(rtmp, rtmp2)
            end do
        end do
        obj%uhat_fit_b = decompose(obj%uhat_f, rtol)

    end

end module
