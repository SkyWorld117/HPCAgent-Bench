! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s2710, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s2710_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D) bind(C, name="tsvc_2_s2710_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(inout) :: b(LEN_1D)
    real(c_double), intent(inout) :: c(LEN_1D)
    real(c_double), intent(in) :: d(LEN_1D)
    real(c_double), intent(in) :: e(LEN_1D)
    real(c_double), intent(in) :: x(LEN_1D)
    integer(c_int64_t) :: i

    do i = 0, (LEN_1D) - 1
        if ((a((i) + 1) > b((i) + 1))) then
            a((i) + 1) = (a((i) + 1) + (b((i) + 1) * d((i) + 1)))
            if ((LEN_1D > 10)) then
                c((i) + 1) = (c((i) + 1) + (d((i) + 1) * d((i) + 1)))
            else
                c((i) + 1) = ((d((i) + 1) * e((i) + 1)) + 1.0_c_double)
            end if
        else
            b((i) + 1) = (a((i) + 1) + (e((i) + 1) * e((i) + 1)))
            if ((x((0) + 1) > 0.0_c_double)) then
                c((i) + 1) = (a((i) + 1) + (d((i) + 1) * d((i) + 1)))
            else
                c((i) + 1) = (c((i) + 1) + (e((i) + 1) * e((i) + 1)))
            end if
        end if
    end do

end subroutine tsvc_2_s2710_fp64
