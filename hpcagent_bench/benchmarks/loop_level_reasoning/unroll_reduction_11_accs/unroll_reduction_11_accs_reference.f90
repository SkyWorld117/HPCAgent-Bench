! Fortran baseline reference for HPCAgent-Bench kernel unroll_reduction_11_accs, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from unroll_reduction_11_accs_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine unroll_reduction_11_accs_fp64(a, out, N) bind(C, name="unroll_reduction_11_accs_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: N
    real(c_double), intent(in) :: a(N)
    real(c_double), intent(inout) :: out(1)

    real(c_double) :: s0
    real(c_double) :: s1
    real(c_double) :: s2
    real(c_double) :: s3
    real(c_double) :: s4
    real(c_double) :: s5
    real(c_double) :: s6
    real(c_double) :: s7
    real(c_double) :: s8
    real(c_double) :: s9
    real(c_double) :: s10
    integer(c_int64_t) :: i
    real(c_double) :: tail
    s0 = 0.0_c_double
    s1 = 0.0_c_double
    s2 = 0.0_c_double
    s3 = 0.0_c_double
    s4 = 0.0_c_double
    s5 = 0.0_c_double
    s6 = 0.0_c_double
    s7 = 0.0_c_double
    s8 = 0.0_c_double
    s9 = 0.0_c_double
    s10 = 0.0_c_double
    i = 0
    do while (((i + 11) <= N))
        s0 = s0 + (a(((i + 0)) + 1))
        s1 = s1 + (a(((i + 1)) + 1))
        s2 = s2 + (a(((i + 2)) + 1))
        s3 = s3 + (a(((i + 3)) + 1))
        s4 = s4 + (a(((i + 4)) + 1))
        s5 = s5 + (a(((i + 5)) + 1))
        s6 = s6 + (a(((i + 6)) + 1))
        s7 = s7 + (a(((i + 7)) + 1))
        s8 = s8 + (a(((i + 8)) + 1))
        s9 = s9 + (a(((i + 9)) + 1))
        s10 = s10 + (a(((i + 10)) + 1))
        i = i + (11)
    end do
    tail = 0.0_c_double
    do while ((i < N))
        tail = tail + (a((i) + 1))
        i = i + (1)
    end do
    out((0) + 1) = (((((((((((s0 + s1) + s2) + s3) + s4) + s5) + s6) + s7) + s8) + s9) + s10) + tail)

end subroutine unroll_reduction_11_accs_fp64
