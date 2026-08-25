! Fortran baseline reference for HPCAgent-Bench kernel mat_scaled_add, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from mat_scaled_add_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine mat_scaled_add_fp64(A, B, M, N, alpha) bind(C, name="mat_scaled_add_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: M
    integer(c_int64_t), value, intent(in) :: N
    real(c_double), intent(in) :: A(N, M)
    real(c_double), intent(inout) :: B(N, M)
    real(c_double), value, intent(in) :: alpha
    integer(c_int64_t) :: i, j

    do i = 0, (M) - 1
        do j = 0, (N) - 1
            B((j) + 1, (i) + 1) = (B((j) + 1, (i) + 1) + (alpha * A((j) + 1, (i) + 1)))
        end do
    end do

end subroutine mat_scaled_add_fp64
