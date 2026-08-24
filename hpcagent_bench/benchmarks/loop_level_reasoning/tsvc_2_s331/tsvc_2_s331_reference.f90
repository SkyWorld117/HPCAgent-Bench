! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s331, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s331_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s331_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s331_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: b(2)
    integer(c_int64_t) :: i
    real(c_double) :: j
    j = (-1)
    j = (-1)
    do i = 0, (LEN_1D) - 1
        if ((a((i) + 1) < 0.0_c_double)) then
            j = i
        end if
    end do
    b((0) + 1) = j

end subroutine tsvc_2_s331_fp64
