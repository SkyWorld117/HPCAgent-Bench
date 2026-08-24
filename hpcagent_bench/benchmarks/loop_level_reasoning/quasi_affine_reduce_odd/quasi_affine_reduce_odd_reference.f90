! Fortran baseline reference for HPCAgent-Bench kernel quasi_affine_reduce_odd, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from quasi_affine_reduce_odd_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine quasi_affine_reduce_odd_fp64(a, out, LEN_1D) bind(C, name="quasi_affine_reduce_odd_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: out(1)
    integer(c_int64_t) :: i

    out((0) + 1) = 0.0_c_double
    do i = 1, (LEN_1D) - 1, 2
        out((0) + 1) = (out((0) + 1) + a((i) + 1))
    end do

end subroutine quasi_affine_reduce_odd_fp64
