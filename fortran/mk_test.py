print(
"""
module sparse_ir_preset
    use sparse_ir
    implicit none

    double precision :: arr(10000)

    contains
    subroutine init()
"""
)
for i in range(100):
    print(
f"""
        arr({1+100*i}:{100+100*i}) = (/ &
""", end =""
    )
    for j in range(100):
        end = ", &\n" if j != 100-1 else ""
        print(12 * " " + "1.d0", end=end)
    print("/)")
print("    end subroutine")
print("end")
