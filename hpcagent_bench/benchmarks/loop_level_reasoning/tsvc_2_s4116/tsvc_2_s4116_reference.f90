! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s4116, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s4116_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s4116_fp64(a, aa, ip, sum_out, LEN_1D, LEN_2D, inc, j) bind(C, name="tsvc_2_s4116_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t), value, intent(in) :: LEN_2D
    integer(c_int64_t), value, intent(in) :: inc
    integer(c_int64_t), value, intent(in) :: j
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(in) :: aa(LEN_2D, 2)
    integer(c_int32_t), intent(in) :: ip(LEN_2D)
    real(c_double), intent(inout) :: sum_out(1)
    integer(c_int64_t) :: i
    real(c_double) :: sum_val
    integer(c_int64_t) :: off
    sum_val = 0.0_c_double
    sum_val = 0.0_c_double
    do i = 0, ((LEN_2D - 1)) - 1
        off = (inc + i)
        sum_val = (sum_val + (a((off) + 1) * aa((INT(ip((i) + 1), c_int64_t)) + 1, ((j - 1)) + 1)))
    end do
    sum_out((0) + 1) = sum_val

end subroutine tsvc_2_s4116_fp64
