! Fortran baseline reference for HPCAgent-Bench kernel halo_broadcast, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from halo_broadcast_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine halo_broadcast_fp64(a, LEN_1D, scale) bind(C, name="halo_broadcast_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), value, intent(in) :: scale
    integer(c_int64_t) :: i

    do i = 1, (LEN_1D) - 1
        a((i) + 1) = ((a((i) + 1) * scale) + a((0) + 1))
    end do

end subroutine halo_broadcast_fp64
