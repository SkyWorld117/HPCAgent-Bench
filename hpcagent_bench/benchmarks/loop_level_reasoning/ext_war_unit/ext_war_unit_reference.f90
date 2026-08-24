! Fortran baseline reference for HPCAgent-Bench kernel ext_war_unit, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from ext_war_unit_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine ext_war_unit_fp64(a, b, LEN_1D) bind(C, name="ext_war_unit_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int64_t) :: i

    do i = 0, ((LEN_1D - 1)) - 1
        a((i) + 1) = (a(((i + 1)) + 1) + b((i) + 1))
    end do

end subroutine ext_war_unit_fp64
