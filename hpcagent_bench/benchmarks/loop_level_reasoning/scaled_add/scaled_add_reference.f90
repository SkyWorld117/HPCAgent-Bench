! Fortran baseline reference for HPCAgent-Bench kernel scaled_add, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from scaled_add_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine scaled_add_fp64(x, y, LEN_1D, alpha) bind(C, name="scaled_add_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: x(LEN_1D)
    real(c_double), intent(inout) :: y(LEN_1D)
    real(c_double), value, intent(in) :: alpha
    integer(c_int64_t) :: i

    do i = 0, (LEN_1D) - 1
        y((i) + 1) = (y((i) + 1) + (alpha * x((i) + 1)))
    end do

end subroutine scaled_add_fp64
