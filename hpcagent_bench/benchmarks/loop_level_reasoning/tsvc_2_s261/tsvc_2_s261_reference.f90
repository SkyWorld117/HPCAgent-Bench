! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s261, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s261_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s261_fp64(a, b, c, d, LEN_1D) bind(C, name="tsvc_2_s261_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    real(c_double), intent(inout) :: c(LEN_1D)
    real(c_double), intent(in) :: d(LEN_1D)
    integer(c_int64_t) :: i
    real(c_double) :: t
    do i = 1, (LEN_1D) - 1
        t = (a((i) + 1) + b((i) + 1))
        a((i) + 1) = (t + c(((i - 1)) + 1))
        c((i) + 1) = (c((i) + 1) * d((i) + 1))
    end do

end subroutine tsvc_2_s261_fp64
