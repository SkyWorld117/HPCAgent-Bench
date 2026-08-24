! Fortran baseline reference for HPCAgent-Bench kernel argmax_with_index, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from argmax_with_index_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C, name="argmax_with_index_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    integer(c_int64_t), intent(inout) :: out_index(1)
    real(c_double), intent(inout) :: out_value(1)
    integer(c_int64_t) :: i
    real(c_double) :: x
    real(c_double) :: idx
    x = a((0) + 1)
    idx = 0
    do i = 1, (LEN_1D) - 1
        if ((a((i) + 1) > x)) then
            x = a((i) + 1)
            idx = i
        end if
    end do
    out_value((0) + 1) = x
    out_index((0) + 1) = idx

end subroutine argmax_with_index_fp64
