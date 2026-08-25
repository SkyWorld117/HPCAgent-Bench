! Fortran baseline reference for HPCAgent-Bench kernel config_select_branch, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from config_select_branch_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine config_select_branch_fp64(out_a, out_b, src, K, LEN_1D) bind(C, name="config_select_branch_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: K
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: out_a(LEN_1D)
    real(c_double), intent(inout) :: out_b(LEN_1D)
    real(c_double), intent(in) :: src(LEN_1D)
    integer(c_int64_t) :: i

    do i = 0, (LEN_1D) - 1
        if ((K > 0)) then
            out_a((i) + 1) = (src((i) + 1) * 2.0_c_double)
        else
            out_b((i) + 1) = (src((i) + 1) + 1.0_c_double)
        end if
    end do

end subroutine config_select_branch_fp64
