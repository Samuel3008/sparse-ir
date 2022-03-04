program main
    use sparse_ir
    use sparse_ir_io
    implicit none

    call test1()

    contains

    subroutine test1()
        type(IR) :: ir_obj
        double precision, parameter :: beta = 1d+4
        double precision, parameter :: wmax = 1.d0

        open(99, file='ir.dat', status='old')
        ir_obj = read_ir(99)
        close(99)
    end

end program