! Fortran baseline reference for HPCAgent-Bench kernel s353_gather_reduction_unroll, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from s353_gather_reduction_unroll_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine s353_gather_reduction_unroll_fp64(a, b, ip, N) bind(C, name="s353_gather_reduction_unroll_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: N
    real(c_double), intent(in) :: a(N)
    real(c_double), intent(inout) :: b(1)
    integer(c_int64_t), intent(in) :: ip(N)

    real(c_double) :: s0
    real(c_double) :: s1
    real(c_double) :: s2
    real(c_double) :: s3
    real(c_double) :: s4
    real(c_double) :: s5
    real(c_double) :: s6
    integer(c_int64_t) :: i
    real(c_double) :: tail
    s0 = 0.0_c_double
    s1 = 0.0_c_double
    s2 = 0.0_c_double
    s3 = 0.0_c_double
    s4 = 0.0_c_double
    s5 = 0.0_c_double
    s6 = 0.0_c_double
    i = 0
    do while (((i + 7) <= N))
        s0 = s0 + (a((ip(((i + 0)) + 1)) + 1))
        s1 = s1 + (a((ip(((i + 1)) + 1)) + 1))
        s2 = s2 + (a((ip(((i + 2)) + 1)) + 1))
        s3 = s3 + (a((ip(((i + 3)) + 1)) + 1))
        s4 = s4 + (a((ip(((i + 4)) + 1)) + 1))
        s5 = s5 + (a((ip(((i + 5)) + 1)) + 1))
        s6 = s6 + (a((ip(((i + 6)) + 1)) + 1))
        i = i + (7)
    end do
    tail = 0.0_c_double
    do while ((i < N))
        tail = tail + (a((ip((i) + 1)) + 1))
        i = i + (1)
    end do
    b((0) + 1) = (((((((s0 + s1) + s2) + s3) + s4) + s5) + s6) + tail)

end subroutine s353_gather_reduction_unroll_fp64
