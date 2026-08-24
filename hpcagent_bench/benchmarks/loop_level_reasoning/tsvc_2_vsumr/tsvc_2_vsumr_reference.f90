! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_vsumr, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_vsumr_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_vsumr_fp64(a, sum_out, LEN_1D) bind(C, name="tsvc_2_vsumr_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: sum_out(1)
    integer(c_int64_t) :: i
    real(c_double) :: s
    s = 0.0_c_double
    s = 0.0_c_double
    do i = 0, (LEN_1D) - 1
        s = (s + a((i) + 1))
    end do
    sum_out((0) + 1) = s

end subroutine tsvc_2_vsumr_fp64
