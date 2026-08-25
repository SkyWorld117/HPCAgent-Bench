! Fortran baseline reference for HPCAgent-Bench kernel argmin_value, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from argmin_value_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine argmin_value_fp64(a, out, LEN_1D) bind(C, name="argmin_value_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: out(1)
    integer(c_int64_t) :: i
    real(c_double) :: x
    x = a((0) + 1)
    do i = 1, (LEN_1D) - 1
        if ((a((i) + 1) < x)) then
            x = a((i) + 1)
        end if
    end do
    out((0) + 1) = x

end subroutine argmin_value_fp64
