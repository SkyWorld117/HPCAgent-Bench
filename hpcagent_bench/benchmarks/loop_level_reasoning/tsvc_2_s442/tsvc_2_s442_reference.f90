! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s442, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s442_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s442_fp64(a, b, c, d, e, indx, LEN_1D) bind(C, name="tsvc_2_s442_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D)
    real(c_double), intent(in) :: d(LEN_1D)
    real(c_double), intent(in) :: e(LEN_1D)
    integer(c_int32_t), intent(in) :: indx(LEN_1D)
    integer(c_int64_t) :: i

    do i = 0, (LEN_1D) - 1
        if ((INT(indx((i) + 1), c_int64_t) == 1)) then
            a((i) + 1) = (a((i) + 1) + (b((i) + 1) * b((i) + 1)))
        else if ((INT(indx((i) + 1), c_int64_t) == 2)) then
            a((i) + 1) = (a((i) + 1) + (c((i) + 1) * c((i) + 1)))
        else if ((INT(indx((i) + 1), c_int64_t) == 3)) then
            a((i) + 1) = (a((i) + 1) + (d((i) + 1) * d((i) + 1)))
        else if ((INT(indx((i) + 1), c_int64_t) == 4)) then
            a((i) + 1) = (a((i) + 1) + (e((i) + 1) * e((i) + 1)))
        end if
    end do

end subroutine tsvc_2_s442_fp64
