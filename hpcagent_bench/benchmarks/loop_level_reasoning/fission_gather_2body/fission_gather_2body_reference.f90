! Fortran baseline reference for HPCAgent-Bench kernel fission_gather_2body, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from fission_gather_2body_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine fission_gather_2body_fp64(a, b, c, e, idx, LEN_1D) bind(C, name="fission_gather_2body_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: b(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D)
    real(c_double), intent(inout) :: e(LEN_1D)
    integer(c_int64_t), intent(in) :: idx(LEN_1D)
    integer(c_int64_t) :: i

    do i = 0, (LEN_1D) - 1
        b((i) + 1) = a((idx((i) + 1)) + 1)
        e((i) + 1) = c((idx((i) + 1)) + 1)
    end do

end subroutine fission_gather_2body_fp64
