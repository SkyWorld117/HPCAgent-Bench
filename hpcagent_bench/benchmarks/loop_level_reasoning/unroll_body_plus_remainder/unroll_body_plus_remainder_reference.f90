! Fortran baseline reference for HPCAgent-Bench kernel unroll_body_plus_remainder, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from unroll_body_plus_remainder_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine unroll_body_plus_remainder_fp64(a, b, N) bind(C, name="unroll_body_plus_remainder_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: N
    real(c_double), intent(in) :: a(N)
    real(c_double), intent(inout) :: b(N)

    real(c_double) :: K
    integer(c_int64_t) :: i
    K = 8
    i = 0
    do while (((i + K) <= N))
        b(((i + 0)) + 1) = (a(((i + 0)) + 1) * a(((i + 0)) + 1))
        b(((i + 1)) + 1) = (a(((i + 1)) + 1) * a(((i + 1)) + 1))
        b(((i + 2)) + 1) = (a(((i + 2)) + 1) * a(((i + 2)) + 1))
        b(((i + 3)) + 1) = (a(((i + 3)) + 1) * a(((i + 3)) + 1))
        b(((i + 4)) + 1) = (a(((i + 4)) + 1) * a(((i + 4)) + 1))
        b(((i + 5)) + 1) = (a(((i + 5)) + 1) * a(((i + 5)) + 1))
        b(((i + 6)) + 1) = (a(((i + 6)) + 1) * a(((i + 6)) + 1))
        b(((i + 7)) + 1) = (a(((i + 7)) + 1) * a(((i + 7)) + 1))
        i = i + (K)
    end do
    do while ((i < N))
        b((i) + 1) = (a((i) + 1) * a((i) + 1))
        i = i + (1)
    end do

end subroutine unroll_body_plus_remainder_fp64
