program main
    use sparse_ir
    implicit none

    type(IR) :: ir_obj
    double precision, parameter :: beta = 1d+4
    double precision, parameter :: wmax = 1.d0

    open(99, file='ir.dat', status='old')
    ir_obj = read_ir(99, beta)
    close(99)

end program