! Fortran baseline reference for HPCAgent-Bench kernel s353_2d_row_unroll_K, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from s353_2d_row_unroll_K_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine s353_2d_row_unroll_k_fp64(a, b, ip, N) bind(C, name="s353_2d_row_unroll_k_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: N
    real(c_double), intent(in) :: a(N, N)
    real(c_double), intent(inout) :: b(N, N)
    integer(c_int64_t), intent(in) :: ip(N)
    integer(c_int64_t) :: j
    integer(c_int64_t) :: i
    i = 0
    do while (((i + 11) <= N))
        do j = 0, (N) - 1
            b((j) + 1, ((i + 0)) + 1) = (a((ip((j) + 1)) + 1, ((i + 0)) + 1) + 1.0_c_double)
            b((j) + 1, ((i + 1)) + 1) = (a((ip((j) + 1)) + 1, ((i + 1)) + 1) + 1.0_c_double)
            b((j) + 1, ((i + 2)) + 1) = (a((ip((j) + 1)) + 1, ((i + 2)) + 1) + 1.0_c_double)
            b((j) + 1, ((i + 3)) + 1) = (a((ip((j) + 1)) + 1, ((i + 3)) + 1) + 1.0_c_double)
            b((j) + 1, ((i + 4)) + 1) = (a((ip((j) + 1)) + 1, ((i + 4)) + 1) + 1.0_c_double)
            b((j) + 1, ((i + 5)) + 1) = (a((ip((j) + 1)) + 1, ((i + 5)) + 1) + 1.0_c_double)
            b((j) + 1, ((i + 6)) + 1) = (a((ip((j) + 1)) + 1, ((i + 6)) + 1) + 1.0_c_double)
            b((j) + 1, ((i + 7)) + 1) = (a((ip((j) + 1)) + 1, ((i + 7)) + 1) + 1.0_c_double)
            b((j) + 1, ((i + 8)) + 1) = (a((ip((j) + 1)) + 1, ((i + 8)) + 1) + 1.0_c_double)
            b((j) + 1, ((i + 9)) + 1) = (a((ip((j) + 1)) + 1, ((i + 9)) + 1) + 1.0_c_double)
            b((j) + 1, ((i + 10)) + 1) = (a((ip((j) + 1)) + 1, ((i + 10)) + 1) + 1.0_c_double)
        end do
        i = i + (11)
    end do
    do while ((i < N))
        do j = 0, (N) - 1
            b((j) + 1, (i) + 1) = (a((ip((j) + 1)) + 1, (i) + 1) + 1.0_c_double)
        end do
        i = i + (1)
    end do

end subroutine s353_2d_row_unroll_k_fp64
