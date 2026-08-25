! Fortran baseline reference for HPCAgent-Bench kernel fuse_stencil_through_transient, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from fuse_stencil_through_transient_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D) bind(C, name="fuse_stencil_through_transient_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: out(LEN_1D)
    integer(c_int64_t) :: i
    real(c_double) :: tmp(LEN_1D)
    do i = 1, ((LEN_1D - 1)) - 1
        tmp((i) + 1) = ((a(((i - 1)) + 1) + a((i) + 1)) + a(((i + 1)) + 1))
    end do
    do i = 1, ((LEN_1D - 2)) - 1
        out((i) + 1) = (tmp((i) + 1) * tmp(((i + 1)) + 1))
    end do

end subroutine fuse_stencil_through_transient_fp64
