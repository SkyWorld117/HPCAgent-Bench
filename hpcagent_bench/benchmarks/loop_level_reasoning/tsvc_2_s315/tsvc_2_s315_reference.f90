! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s315, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s315_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s315_fp64(a, result, LEN_1D) bind(C, name="tsvc_2_s315_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(inout) :: result(1)
    integer(c_int64_t) :: i
    real(c_double) :: x
    real(c_double) :: index
    do i = 0, (LEN_1D) - 1
        a((i) + 1) = MODULO((i * 7), LEN_1D)
    end do
    x = a((0) + 1)
    index = 0
    do i = 0, (LEN_1D) - 1
        if ((a((i) + 1) > x)) then
            x = a((i) + 1)
            index = i
        end if
    end do
    a((0) + 1) = (x + index)
    result((0) + 1) = a((0) + 1)

end subroutine tsvc_2_s315_fp64
