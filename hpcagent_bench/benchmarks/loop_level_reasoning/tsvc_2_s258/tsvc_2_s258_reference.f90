! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s258, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s258_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s258_fp64(a, aa, b, c, d, e, LEN_2D) bind(C, name="tsvc_2_s258_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(in) :: a(LEN_2D)
    real(c_double), intent(in) :: aa(LEN_2D, 1)
    real(c_double), intent(inout) :: b(LEN_2D)
    real(c_double), intent(in) :: c(LEN_2D)
    real(c_double), intent(in) :: d(LEN_2D)
    real(c_double), intent(inout) :: e(LEN_2D)
    integer(c_int64_t) :: i
    real(c_double) :: s
    s = 0.0_c_double
    do i = 0, (LEN_2D) - 1
        if ((a((i) + 1) > 0.0_c_double)) then
            s = (d((i) + 1) * d((i) + 1))
        end if
        b((i) + 1) = ((s * c((i) + 1)) + d((i) + 1))
        e((i) + 1) = ((s + 1.0_c_double) * aa((i) + 1, (0) + 1))
    end do

end subroutine tsvc_2_s258_fp64
