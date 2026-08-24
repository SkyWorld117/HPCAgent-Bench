! Fortran baseline reference for HPCAgent-Bench kernel ext_break_capture, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from ext_break_capture_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine ext_break_capture_fp64(a, out_index, out_value, K, LEN_1D) bind(C, name="ext_break_capture_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: K
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    integer(c_int64_t), intent(inout) :: out_index(1)
    real(c_double), intent(inout) :: out_value(1)
    integer(c_int64_t) :: i

    out_index((0) + 1) = (-1)
    out_value((0) + 1) = (-(1.0_c_double))
    do i = 0, (LEN_1D) - 1
        if ((a((i) + 1) > K)) then
            out_index((0) + 1) = i
            out_value((0) + 1) = a((i) + 1)
            exit
        end if
    end do

end subroutine ext_break_capture_fp64
