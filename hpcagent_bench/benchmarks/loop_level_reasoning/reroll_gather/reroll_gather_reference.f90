! Fortran baseline reference for HPCAgent-Bench kernel reroll_gather, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from reroll_gather_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine reroll_gather_fp64(a, b, ip, LEN_1D) bind(C, name="reroll_gather_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int64_t), intent(in) :: ip(LEN_1D)
    integer(c_int64_t) :: i

    do i = 0, ((LEN_1D - 6)) - 1, 7
        a((i) + 1) = (a((i) + 1) + (b((ip((i) + 1)) + 1) * 2.0_c_double))
        a(((i + 1)) + 1) = (a(((i + 1)) + 1) + (b((ip(((i + 1)) + 1)) + 1) * 2.0_c_double))
        a(((i + 2)) + 1) = (a(((i + 2)) + 1) + (b((ip(((i + 2)) + 1)) + 1) * 2.0_c_double))
        a(((i + 3)) + 1) = (a(((i + 3)) + 1) + (b((ip(((i + 3)) + 1)) + 1) * 2.0_c_double))
        a(((i + 4)) + 1) = (a(((i + 4)) + 1) + (b((ip(((i + 4)) + 1)) + 1) * 2.0_c_double))
        a(((i + 5)) + 1) = (a(((i + 5)) + 1) + (b((ip(((i + 5)) + 1)) + 1) * 2.0_c_double))
        a(((i + 6)) + 1) = (a(((i + 6)) + 1) + (b((ip(((i + 6)) + 1)) + 1) * 2.0_c_double))
    end do

end subroutine reroll_gather_fp64
