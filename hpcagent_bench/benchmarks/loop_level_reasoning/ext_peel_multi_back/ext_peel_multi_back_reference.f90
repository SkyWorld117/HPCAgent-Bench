! Fortran baseline reference for HPCAgent-Bench kernel ext_peel_multi_back, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from ext_peel_multi_back_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine ext_peel_multi_back_fp64(a, b, LEN_1D) bind(C, name="ext_peel_multi_back_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int64_t) :: i

    do i = 0, (LEN_1D) - 1
        a((i) + 1) = (b((i) + 1) * 2.0_c_double)
        if ((i == (LEN_1D - 1))) then
            a(((LEN_1D - 2)) + 1) = (a(((LEN_1D - 2)) + 1) + 1.0_c_double)
        else if ((i == (LEN_1D - 2))) then
            a(((LEN_1D - 3)) + 1) = (a(((LEN_1D - 3)) + 1) + 1.0_c_double)
        end if
    end do

end subroutine ext_peel_multi_back_fp64
