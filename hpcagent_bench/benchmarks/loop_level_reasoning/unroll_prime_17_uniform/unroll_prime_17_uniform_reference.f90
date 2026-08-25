! Fortran baseline reference for HPCAgent-Bench kernel unroll_prime_17_uniform, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from unroll_prime_17_uniform_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine unroll_prime_17_uniform_fp64(a, b, N) bind(C, name="unroll_prime_17_uniform_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: N
    real(c_double), intent(in) :: a(N)
    real(c_double), intent(inout) :: b(N)

    integer(c_int64_t) :: i
    i = 0
    do while (((i + 17) <= N))
        b(((i + 0)) + 1) = (a(((i + 0)) + 1) + 1.0_c_double)
        b(((i + 1)) + 1) = (a(((i + 1)) + 1) + 1.0_c_double)
        b(((i + 2)) + 1) = (a(((i + 2)) + 1) + 1.0_c_double)
        b(((i + 3)) + 1) = (a(((i + 3)) + 1) + 1.0_c_double)
        b(((i + 4)) + 1) = (a(((i + 4)) + 1) + 1.0_c_double)
        b(((i + 5)) + 1) = (a(((i + 5)) + 1) + 1.0_c_double)
        b(((i + 6)) + 1) = (a(((i + 6)) + 1) + 1.0_c_double)
        b(((i + 7)) + 1) = (a(((i + 7)) + 1) + 1.0_c_double)
        b(((i + 8)) + 1) = (a(((i + 8)) + 1) + 1.0_c_double)
        b(((i + 9)) + 1) = (a(((i + 9)) + 1) + 1.0_c_double)
        b(((i + 10)) + 1) = (a(((i + 10)) + 1) + 1.0_c_double)
        b(((i + 11)) + 1) = (a(((i + 11)) + 1) + 1.0_c_double)
        b(((i + 12)) + 1) = (a(((i + 12)) + 1) + 1.0_c_double)
        b(((i + 13)) + 1) = (a(((i + 13)) + 1) + 1.0_c_double)
        b(((i + 14)) + 1) = (a(((i + 14)) + 1) + 1.0_c_double)
        b(((i + 15)) + 1) = (a(((i + 15)) + 1) + 1.0_c_double)
        b(((i + 16)) + 1) = (a(((i + 16)) + 1) + 1.0_c_double)
        i = i + (17)
    end do
    do while ((i < N))
        b((i) + 1) = (a((i) + 1) + 1.0_c_double)
        i = i + (1)
    end do

end subroutine unroll_prime_17_uniform_fp64
