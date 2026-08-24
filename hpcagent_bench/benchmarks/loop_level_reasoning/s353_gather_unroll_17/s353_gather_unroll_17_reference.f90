! Fortran baseline reference for HPCAgent-Bench kernel s353_gather_unroll_17, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from s353_gather_unroll_17_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine s353_gather_unroll_17_fp64(a, b, ip, N) bind(C, name="s353_gather_unroll_17_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: N
    real(c_double), intent(in) :: a(N)
    real(c_double), intent(inout) :: b(N)
    integer(c_int64_t), intent(in) :: ip(N)

    integer(c_int64_t) :: i
    i = 0
    do while (((i + 17) <= N))
        b(((i + 0)) + 1) = (a((ip(((i + 0)) + 1)) + 1) + 1.0_c_double)
        b(((i + 1)) + 1) = (a((ip(((i + 1)) + 1)) + 1) + 1.0_c_double)
        b(((i + 2)) + 1) = (a((ip(((i + 2)) + 1)) + 1) + 1.0_c_double)
        b(((i + 3)) + 1) = (a((ip(((i + 3)) + 1)) + 1) + 1.0_c_double)
        b(((i + 4)) + 1) = (a((ip(((i + 4)) + 1)) + 1) + 1.0_c_double)
        b(((i + 5)) + 1) = (a((ip(((i + 5)) + 1)) + 1) + 1.0_c_double)
        b(((i + 6)) + 1) = (a((ip(((i + 6)) + 1)) + 1) + 1.0_c_double)
        b(((i + 7)) + 1) = (a((ip(((i + 7)) + 1)) + 1) + 1.0_c_double)
        b(((i + 8)) + 1) = (a((ip(((i + 8)) + 1)) + 1) + 1.0_c_double)
        b(((i + 9)) + 1) = (a((ip(((i + 9)) + 1)) + 1) + 1.0_c_double)
        b(((i + 10)) + 1) = (a((ip(((i + 10)) + 1)) + 1) + 1.0_c_double)
        b(((i + 11)) + 1) = (a((ip(((i + 11)) + 1)) + 1) + 1.0_c_double)
        b(((i + 12)) + 1) = (a((ip(((i + 12)) + 1)) + 1) + 1.0_c_double)
        b(((i + 13)) + 1) = (a((ip(((i + 13)) + 1)) + 1) + 1.0_c_double)
        b(((i + 14)) + 1) = (a((ip(((i + 14)) + 1)) + 1) + 1.0_c_double)
        b(((i + 15)) + 1) = (a((ip(((i + 15)) + 1)) + 1) + 1.0_c_double)
        b(((i + 16)) + 1) = (a((ip(((i + 16)) + 1)) + 1) + 1.0_c_double)
        i = i + (17)
    end do
    do while ((i < N))
        b((i) + 1) = (a((ip((i) + 1)) + 1) + 1.0_c_double)
        i = i + (1)
    end do

end subroutine s353_gather_unroll_17_fp64
