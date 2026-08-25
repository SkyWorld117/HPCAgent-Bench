! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s211, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s211_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s211_fp64(a, b, c, d, e, LEN_1D) bind(C, name="tsvc_2_s211_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(inout) :: b(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D)
    real(c_double), intent(in) :: d(LEN_1D)
    real(c_double), intent(in) :: e(LEN_1D)
    integer(c_int64_t) :: i

    do i = 1, ((LEN_1D - 1)) - 1
        a((i) + 1) = (b(((i - 1)) + 1) + (c((i) + 1) * d((i) + 1)))
        b((i) + 1) = (b(((i + 1)) + 1) - (e((i) + 1) * d((i) + 1)))
    end do

end subroutine tsvc_2_s211_fp64
