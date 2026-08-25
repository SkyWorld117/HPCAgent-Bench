! Fortran baseline reference for HPCAgent-Bench kernel fission_dep_sym_offset, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from fission_dep_sym_offset_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine fission_dep_sym_offset_fp64(a, b, x, y, z, K, LEN_1D) bind(C, name="fission_dep_sym_offset_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: K
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(inout) :: b(LEN_1D)
    real(c_double), intent(in) :: x(LEN_1D)
    real(c_double), intent(in) :: y(LEN_1D)
    real(c_double), intent(in) :: z(LEN_1D)
    integer(c_int64_t) :: i

    do i = K, (LEN_1D) - 1
        a((i) + 1) = (a(((i - K)) + 1) + x((i) + 1))
        b((i) + 1) = (y((i) + 1) * z((i) + 1))
    end do

end subroutine fission_dep_sym_offset_fp64
