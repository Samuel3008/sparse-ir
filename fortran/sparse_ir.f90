module sparse_ir
    implicit none

    type DecomposedMatrix
        double precision, allocatable :: inv_s
        complex(kind(0d0)), allocatable :: ut(:, :), v(:, :)
    end type

    type IR
        integer::size
        complex(kind(0d0)), allocatable :: u(:, :)
        complex(kind(0d0)), allocatable :: uhat(:, :)
    end type

    contains

    function decompose_matrix(mat) result(dmat)
        complex(kind(0d0)), intent(in) :: mat(:, :)

        type(DecomposedMatrix)::dmat
        !allocate(dmat%u())

    end function

    function read(unit) result(obj)
        !character (len=*), intent (in) :: filename
        integer, intent (in) :: unit

        type(IR)::obj
        allocate(obj%u(obj%size, 10))
        allocate(obj%uhat(obj%size, 10))
    end function

end module
