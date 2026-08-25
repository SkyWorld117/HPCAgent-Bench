! Fortran baseline reference for HPCAgent-Bench kernel scan_conditional, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from scan_conditional_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine scan_conditional_fp64(delta, mask, out, LEN_1D) bind(C, name="scan_conditional_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: delta(LEN_1D)
    integer(c_int64_t), intent(in) :: mask(LEN_1D)
    real(c_double), intent(inout) :: out(LEN_1D)
    integer(c_int64_t) :: i

    do i = 1, (LEN_1D) - 1
        if ((mask((i) + 1) > 0)) then
            out((i) + 1) = (out(((i - 1)) + 1) + delta((i) + 1))
        else
            out((i) + 1) = out(((i - 1)) + 1)
        end if
    end do

end subroutine scan_conditional_fp64
