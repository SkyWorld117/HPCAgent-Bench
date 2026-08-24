! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s231, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s231_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s231_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s231_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, j

    do i = 0, (LEN_2D) - 1
        do j = 1, (LEN_2D) - 1
            aa((i) + 1, (j) + 1) = (aa((i) + 1, ((j - 1)) + 1) + bb((i) + 1, (j) + 1))
        end do
    end do

end subroutine tsvc_2_s231_fp64
