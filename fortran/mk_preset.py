import argparse
import sparse_ir

def run():
    nlambda_list  = [1, 2, 3, 4, 5]
    ndigit_list  = [10]


    print(
"""\
module sparse_ir_preset
    use sparse_ir
    implicit none
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
            print_data(nlambda, ndigit)

    print("end")

def print_data(nlambda, ndigit):
    lambda_ = 10.0 ** nlambda
    eps = 1/10.0 ** ndigit
    kernel = sparse_ir.LogisticKernel(lambda_)
    sve_result = sparse_ir.compute_sve(kernel, eps)
    basis_f = sparse_ir.IRBasis("F", lambda_, eps, kernel=kernel, sve_result=sve_result)
    basis_b = sparse_ir.IRBasis("B", lambda_, eps, kernel=kernel, sve_result=sve_result)
    smpl_tau = sparse_ir.TauSampling(basis_f)
    smpl_matsu_f = sparse_ir.MatsubaraSampling(basis_f)
    smpl_matsu_b = sparse_ir.MatsubaraSampling(basis_b)

    size = basis_f.size
    ntau = smpl_tau.sampling_points.size
    nfreq_f = smpl_matsu_f.sampling_points.size
    nfreq_b = smpl_matsu_b.sampling_points.size
    print(
f"""
    function mk_nlambda{nlambda}_ndigit{ndigit}(beta) result(obj)
        double precision, intent(in) :: beta
        type(IR) :: obj
        double precision, allocatable :: s(:,:), tau(:,:)
        complex(kind(0d0)), allocatable :: u(:, :), uhat_f(:, :), uhat_b(:, :)
        double precision, allocatable :: u_reduced(:, :), uhat_f_reduced(:, :), uhat_b_reduced(:, :)
        integer, allocatable :: freq_f(:,:), freq_b(:,:)
        integer, parameter :: size = {size}, ntau = {ntau}, nfreq_f = {nfreq_f}, nfreq_b = {nfreq_b}, nlambda = {nlambda}, ndigit = {ndigit}
        integer, parameter :: ntau_reduced = ntau/2+1, nfreq_f_reduced = nfreq_f/2+1, nfreq_b_reduced = nfreq_b/2+1
        double precision, parameter :: lambda = 10.d0 ** nlambda, eps = 1/10.d0**ndigit

        integer :: itau, l, ifreq

        allocate(s(size, 1))
        allocate(u(ntau, size))
        allocate(uhat_f(nfreq_f, size))
        allocate(uhat_b(nfreq_b, size))
        allocate(freq_f(nfreq_f, 1))
        allocate(freq_b(nfreq_b, 1))

        allocate(u_reduced(ntau_reduced, size))
        allocate(uhat_f_reduced(nfreq_f_reduced, size))
        allocate(uhat_b_reduced(nfreq_b_reduced, size))

        s = generator_s_nlambda{nlambda}_ndigit{ndigit}()
        tau = generator_tau_nlambda{nlambda}_ndigit{ndigit}()
        freq_f = generator_freq_f_nlambda{nlambda}_ndigit{ndigit}()
        freq_b = generator_freq_b_nlambda{nlambda}_ndigit{ndigit}()
        u_reduced = generator_u_reduced_nlambda{nlambda}_ndigit{ndigit}()
        uhat_f_reduced = generator_uhatf_reduced_nlambda{nlambda}_ndigit{ndigit}()
        uhat_b_reduced = generator_uhatb_reduced_nlambda{nlambda}_ndigit{ndigit}()

        ! Use the fact U_l(tau) is even/odd for even/odd l-1.
        do l = 1, size
            do itau = 1, ntau_reduced
                u(itau, l) = u_reduced(itau, l)
                u(ntau-itau+1, l) = (-1)**(l-1) * u_reduced(itau, l)
            end do
        end do

        ! Use the fact U^F_l(iv) is pure imaginary/real for even/odd l-1.
        do l = 1, size, 2
            do ifreq = 1, nfreq_f_reduced
                uhat_f(ifreq, l) = dcmplx(0.0, uhat_f_reduced(ifreq, l))
            end do
        end do
        do l = 2, size, 2
            do ifreq = 1, nfreq_f_reduced
                uhat_f(ifreq, l) = dcmplx(uhat_f_reduced(ifreq, l), 0.0)
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
                uhat_b(ifreq, l) = dcmplx(uhat_b_reduced(ifreq, l), 0.0d0)
            end do
        end do
        do l = 2, size, 2
            do ifreq = 1, nfreq_b_reduced
                uhat_b(ifreq, l) = dcmplx(0.0d0, uhat_b_reduced(ifreq, l))
            end do
        end do
        do l = 1, size
            do ifreq = 1, nfreq_b
                uhat_b(nfreq_b-ifreq+1, l) = conjg(uhat_b(ifreq, l))
            end do
        end do

        call init_ir(obj, beta, lambda, eps, s(:,1), tau(:,1), freq_f(:,1), freq_b(:,1), u, uhat_f, uhat_b, 1d-20)

        deallocate(s, u, uhat_f, uhat_b, freq_f, freq_b)
        deallocate(u_reduced, uhat_f_reduced, uhat_b_reduced)
    end
"""
    )

    print_real_matrix_generator(basis_f.s[:,None], f"generator_s_nlambda{nlambda}_ndigit{ndigit}")
    print_real_matrix_generator(smpl_tau.sampling_points[:,None], f"generator_tau_nlambda{nlambda}_ndigit{ndigit}")
    print_int_matrix_generator(smpl_matsu_f.sampling_points[:,None], f"generator_freq_f_nlambda{nlambda}_ndigit{ndigit}")
    print_int_matrix_generator(smpl_matsu_b.sampling_points[:,None], f"generator_freq_b_nlambda{nlambda}_ndigit{ndigit}")

    ntau_reduced = ntau //2 + 1
    print_real_matrix_generator(smpl_tau.matrix.a[0:ntau_reduced,:], f"generator_u_reduced_nlambda{nlambda}_ndigit{ndigit}")
    print_real_matrix_generator(
        smpl_matsu_f.matrix.a.real + smpl_matsu_f.matrix.a.imag, f"generator_uhatf_reduced_nlambda{nlambda}_ndigit{ndigit}")
    print_real_matrix_generator(
        smpl_matsu_b.matrix.a.real + smpl_matsu_b.matrix.a.imag, f"generator_uhatb_reduced_nlambda{nlambda}_ndigit{ndigit}")


def print_real_matrix_generator(matrix, func_name):
    """ Print vector data generator"""
    n, m  = matrix.shape
    print(
f"""
    function {func_name}() result(obj)
        double precision, allocatable :: obj(:,:)
"""
    )
    print(8*" " + f"integer, parameter :: n={n}, m={m}")
    print(8*" " + f"allocate(obj({n}, {m}))")
    for j in range(m):
        for i in range(n):
            print(8*" " + f"obj({i+1},{j+1}) = {matrix[i,j]:.16e}".replace('e', 'd'))
    print(
f"""
    end
"""
    )

def print_int_matrix_generator(matrix, func_name):
    """ Print vector data generator"""
    n, m  = matrix.shape
    print(
f"""
    function {func_name}() result(obj)
        integer, allocatable :: obj(:,:)
"""
    )
    print(8*" " + f"integer, parameter :: n={n}, m={m}")
    print(8*" " + f"allocate(obj({n}, {m}))")
    for j in range(m):
        for i in range(n):
            print(8*" " + f"obj({i+1},{j+1}) = {matrix[i,j]}")
    print(
f"""
    end
"""
    )


if __name__ == '__main__':
    run()
