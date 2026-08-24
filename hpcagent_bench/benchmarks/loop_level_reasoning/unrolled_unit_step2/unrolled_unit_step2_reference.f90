! Fortran baseline reference for HPCAgent-Bench kernel unrolled_unit_step2, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from unrolled_unit_step2_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine unrolled_unit_step2_fp64(a, b, M) bind(C, name="unrolled_unit_step2_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: M
    real(c_double), intent(inout) :: a(M)
    real(c_double), intent(in) :: b(M)
    integer(c_int64_t) :: i

    do i = 0, (M) - 1, 2
        a((i) + 1) = (b((i) + 1) * 2.0_c_double)
        a(((i + 1)) + 1) = (b(((i + 1)) + 1) * 2.0_c_double)
    end do

end subroutine unrolled_unit_step2_fp64
