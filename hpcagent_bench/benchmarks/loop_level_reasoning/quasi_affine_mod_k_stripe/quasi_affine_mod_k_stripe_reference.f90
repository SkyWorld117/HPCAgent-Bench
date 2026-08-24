! Fortran baseline reference for HPCAgent-Bench kernel quasi_affine_mod_k_stripe, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from quasi_affine_mod_k_stripe_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine quasi_affine_mod_k_stripe_fp64(a, b, c, K, LEN_1D) bind(C, name="quasi_affine_mod_k_stripe_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: K
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D)
    integer(c_int64_t) :: i

    do i = 0, (LEN_1D) - 1
        if ((MODULO(i, K) == 0)) then
            a((i) + 1) = (b((i) + 1) * 2.0_c_double)
        else
            a((i) + 1) = c((i) + 1)
        end if
    end do

end subroutine quasi_affine_mod_k_stripe_fp64
