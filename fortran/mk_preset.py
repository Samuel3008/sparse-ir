import sparse_ir
import numpy as np


class BasisInfo:
    def __init__(self, nlambda: int, ndigit: int) -> None:
        lambda_ = 10.0 ** nlambda
        eps = 1/10.0 ** ndigit
        kernel = sparse_ir.LogisticKernel(lambda_)
        sve_result = sparse_ir.compute_sve(kernel, eps)

        self.basis_f = sparse_ir.IRBasis("F", lambda_, eps, kernel=kernel, sve_result=sve_result)
        self.basis_b = sparse_ir.IRBasis("B", lambda_, eps, kernel=kernel, sve_result=sve_result)
        self.smpl_tau = sparse_ir.TauSampling(self.basis_f)
        self.smpl_matsu_f = sparse_ir.MatsubaraSampling(self.basis_f)
        self.smpl_matsu_b = sparse_ir.MatsubaraSampling(self.basis_b)

        self.s = self.basis_f.s
        self.size = self.s.size
        self.tau = self.smpl_tau.sampling_points
        self.freq_f = self.smpl_matsu_f.sampling_points
        self.freq_b = self.smpl_matsu_b.sampling_points
        self.u = self.smpl_tau.matrix.a
        self.uhat_f = self.smpl_matsu_f.matrix.a
        self.uhat_b = self.smpl_matsu_b.matrix.a

        self.ntau = self.tau.size
        self.nfreq_f = self.freq_f.size
        self.nfreq_b = self.freq_b.size

        self.ntau_reduced = self.ntau //2 + 1
        self.nfreq_f_reduced = self.nfreq_f //2 + 1
        self.nfreq_b_reduced = self.nfreq_b //2 + 1


def _to_str(shape):
    return ",".join(map(str, shape))


def run():
    #nlambda_list  = [1, 2, 3, 4, 5]
    #ndigit_list  = [10]
    nlambda_list  = [4]
    ndigit_list  = [10]

    bases = {}

    for nlambda in nlambda_list:
        for ndigit in ndigit_list:
            #print(f"nlambda {nlambda}, ndigit {ndigit}")
            bases[(nlambda, ndigit)] = BasisInfo(nlambda, ndigit)

    print(
"""\
module sparse_ir_preset
    use sparse_ir
    implicit none
""")
    for nlambda in nlambda_list:
        for ndigit in ndigit_list:
            b = bases[(nlambda, ndigit)]
            sig = f"nlambda{nlambda}_ndigit{ndigit}"
            print(
f"""\
    double precision :: s_{sig}({b.size})
    double precision :: tau_{sig}({b.ntau})
    integer :: freq_f_{sig}({b.nfreq_f})
    integer :: freq_b_{sig}({b.nfreq_b})
    double precision :: u_r_{sig}({b.ntau_reduced} * {b.size})
    double precision :: uhat_f_r_{sig}({b.nfreq_f_reduced} * {b.size})
    double precision :: uhat_b_r_{sig}({b.nfreq_b_reduced} * {b.size})
"""
            )

    print(
"""\
    contains

    function mk_ir_preset(nlambda, ndigit, beta) result(obj)
        integer, intent(in) :: nlambda, ndigit
        double precision, intent(in) :: beta
        type(IR) :: obj
""")

    for nlambda in nlambda_list:
        for ndigit in ndigit_list:
            print(
f"""\
        if (nlambda == {nlambda} .and. ndigit == {ndigit}) then
            obj = mk_nlambda{nlambda}_ndigit{ndigit}(beta)
            return
        end if
"""
            )


    print(
"""\
        stop "Invalid parameters"
    end
"""
            )

    for nlambda in nlambda_list:
        for ndigit in ndigit_list:
            print_data(nlambda, ndigit, bases[(nlambda, ndigit)])

    print("end")


def print_data(nlambda, ndigit, b):
    sig = f"nlambda{nlambda}_ndigit{ndigit}"
    size = b.size
    print(
f"""
    function mk_nlambda{nlambda}_ndigit{ndigit}(beta) result(obj)
        double precision, intent(in) :: beta
        type(IR) :: obj
        complex(kind(0d0)), allocatable :: u(:, :), uhat_f(:, :), uhat_b(:, :)
        integer, parameter :: size = {b.size}, ntau = {b.ntau}, nfreq_f = {b.nfreq_f}, nfreq_b = {b.nfreq_b}, nlambda = {nlambda}, ndigit = {ndigit}
        integer, parameter :: ntau_reduced = ntau/2+1, nfreq_f_reduced = nfreq_f/2+1, nfreq_b_reduced = nfreq_b/2+1
        double precision, parameter :: lambda = 1.d1 ** nlambda, eps = 1/1.d1**ndigit

        integer :: itau, l, ifreq
""")
    for varname in ["s", "tau", "freq_f", "freq_b", "u_r", "uhat_f_r", "uhat_b_r"]:
        print(8*" ", f"call init_{varname}_{sig}()")

    print(
f"""
        allocate(u(ntau, size))
        allocate(uhat_f(nfreq_f, size))
        allocate(uhat_b(nfreq_b, size))

        ! Use the fact U_l(tau) is even/odd for even/odd l-1.
        do l = 1, size
            do itau = 1, ntau_reduced
                u(itau, l) = u_r_{sig}(itau + {b.ntau_reduced}*(l-1))
                u(ntau-itau+1, l) = (-1)**(l-1) * u_r_{sig}(itau + {b.ntau_reduced}*(l-1))
            end do
        end do

        ! Use the fact U^F_l(iv) is pure imaginary/real for even/odd l-1.
        do l = 1, size, 2
            do ifreq = 1, nfreq_f_reduced
                uhat_f(ifreq, l) = cmplx(0.0, uhat_f_r_{sig}(ifreq + {b.nfreq_f_reduced}*(l-1)), kind(0d0))
            end do
        end do
        do l = 2, size, 2
            do ifreq = 1, nfreq_f_reduced
                uhat_f(ifreq, l) = cmplx(uhat_f_r_{sig}(ifreq + {b.nfreq_f_reduced}*(l-1)), 0.0, kind(0d0))
            end do
        end do
        do l = 1, size
            do ifreq = 1, nfreq_f
                uhat_f(nfreq_f-ifreq+1, l) = conjg(uhat_f(ifreq, l))
            end do
        end do

        ! Use the fact U^B_l(iv) is pure real/imaginary for even/odd l-1
        do l = 1, size, 2
            do ifreq = 1, nfreq_b_reduced
                uhat_b(ifreq, l) = cmplx(uhat_b_r_{sig}(ifreq + {b.nfreq_b_reduced}*(l-1)), 0.0d0, kind(0d0))
            end do
        end do
        do l = 2, size, 2
            do ifreq = 1, nfreq_b_reduced
                uhat_b(ifreq, l) = cmplx(0.0d0, uhat_b_r_{sig}(ifreq + {b.nfreq_b_reduced}*(l-1)), kind(0d0))
            end do
        end do
        do l = 1, size
            do ifreq = 1, nfreq_b
                uhat_b(nfreq_b-ifreq+1, l) = conjg(uhat_b(ifreq, l))
            end do
        end do

        call init_ir(obj, beta, lambda, eps,&
            s_{sig}, tau_{sig},&
            freq_f_{sig}, freq_b_{sig},&
            u, uhat_f, uhat_b, 1d-20)

        deallocate(u, uhat_f, uhat_b)
    end
"""
    )

    print_real_data(b.s, f"s_{sig}")
    print_real_data(b.tau, f"tau_{sig}")
    print_int_data(b.freq_f, f"freq_f_{sig}")
    print_int_data(b.freq_b, f"freq_b_{sig}")

    ntau_reduced = b.ntau_reduced
    print_real_data(b.u[0:ntau_reduced,:], f"u_r_{sig}")
    print_real_data((b.uhat_f.real + b.uhat_f.imag)[0:b.nfreq_f_reduced,:], f"uhat_f_r_{sig}")
    print_real_data((b.uhat_b.real + b.uhat_b.imag)[0:b.nfreq_b_reduced,:], f"uhat_b_r_{sig}")


def print_real_data(vec, var_name):
    """ Print array data generator"""
    vec = vec.T.ravel()
    n = vec.size
    print(
f"""
    subroutine init_{var_name}()
"""
    )
    for i in range(n):
        print(8*" " + f"{var_name}({i+1}) = " + f"{vec[i]:.16e}".replace('e', 'd'))
    print(
f"""
    end subroutine
"""
    )

def print_int_data(vec, var_name):
    """ Print array data generator"""
    vec = np.asfortranarray(vec)
    vec = vec.T.ravel()
    n = vec.size
    print(
f"""
    subroutine init_{var_name}()
"""
    )
    for i in range(n):
        print(8*" " + f"{var_name}({i+1}) = {vec[i]}")
    print(
f"""
    end subroutine
"""
    )


if __name__ == '__main__':
    run()
