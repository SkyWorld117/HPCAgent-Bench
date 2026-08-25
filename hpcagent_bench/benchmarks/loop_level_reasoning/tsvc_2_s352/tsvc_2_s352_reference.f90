! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s352, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s352_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s352_fp64(a, b, c, LEN_1D) bind(C, name="tsvc_2_s352_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    real(c_double), intent(inout) :: c(2)
    integer(c_int64_t) :: i
    real(c_double) :: dot
    dot = 0.0_c_double
    dot = 0.0_c_double
    do i = 0, ((LEN_1D - 4)) - 1, 5
        dot = (dot + (((((a((i) + 1) * b((i) + 1)) + (a(((i + 1)) + 1) * b(((i + 1)) + 1))) + (a(((i + 2)) + 1) * &
        &b(((i + 2)) + 1))) + (a(((i + 3)) + 1) * b(((i + 3)) + 1))) + (a(((i + 4)) + 1) * b(((i + 4)) + 1))))
    end do
    c((0) + 1) = dot

end subroutine tsvc_2_s352_fp64
