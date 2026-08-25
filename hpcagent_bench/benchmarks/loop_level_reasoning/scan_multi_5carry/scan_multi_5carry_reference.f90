! Fortran baseline reference for HPCAgent-Bench kernel scan_multi_5carry, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from scan_multi_5carry_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine scan_multi_5carry_fp64(acc, delta, LEN_1D) bind(C, name="scan_multi_5carry_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: acc(LEN_1D, 5)
    real(c_double), intent(in) :: delta(LEN_1D, 5)
    integer(c_int64_t) :: i

    do i = 1, (LEN_1D) - 1
        acc((i) + 1, (0) + 1) = (acc(((i - 1)) + 1, (0) + 1) + delta((i) + 1, (0) + 1))
        acc((i) + 1, (1) + 1) = (acc(((i - 1)) + 1, (1) + 1) + delta((i) + 1, (1) + 1))
        acc((i) + 1, (2) + 1) = (acc(((i - 1)) + 1, (2) + 1) + delta((i) + 1, (2) + 1))
        acc((i) + 1, (3) + 1) = (acc(((i - 1)) + 1, (3) + 1) + delta((i) + 1, (3) + 1))
        acc((i) + 1, (4) + 1) = (acc(((i - 1)) + 1, (4) + 1) + delta((i) + 1, (4) + 1))
    end do

end subroutine scan_multi_5carry_fp64
