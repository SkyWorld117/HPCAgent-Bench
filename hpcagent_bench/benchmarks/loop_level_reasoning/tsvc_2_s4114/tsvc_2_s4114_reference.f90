! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s4114, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s4114_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s4114_fp64(a, b, c, d_, ip, LEN_1D, n1) bind(C, name="tsvc_2_s4114_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t), value, intent(in) :: n1
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D)
    real(c_double), intent(in) :: d_(LEN_1D)
    integer(c_int32_t), intent(in) :: ip(LEN_1D)
    integer(c_int64_t) :: i
    integer(c_int64_t) :: k
    do i = (n1 - 1), (LEN_1D) - 1
        k = INT(ip((i) + 1), c_int64_t)
        a((i) + 1) = (b((i) + 1) + (c((((LEN_1D - k) - 1)) + 1) * d_((i) + 1)))
    end do

end subroutine tsvc_2_s4114_fp64
