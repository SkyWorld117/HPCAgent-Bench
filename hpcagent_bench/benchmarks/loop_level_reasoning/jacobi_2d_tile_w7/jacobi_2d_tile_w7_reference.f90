! Fortran baseline reference for HPCAgent-Bench kernel jacobi_2d_tile_w7, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from jacobi_2d_tile_w7_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine jacobi_2d_tile_w7_fp64(A, B, N, TSTEPS) bind(C, name="jacobi_2d_tile_w7_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: N
    integer(c_int64_t), value, intent(in) :: TSTEPS
    real(c_double), intent(inout) :: A(N, N)
    real(c_double), intent(inout) :: B(N, N)
    integer(c_int64_t) :: i, ii, j, jj, t
    integer(c_int64_t) :: W
    W = 7
    do t = 0, (TSTEPS) - 1
        do ii = 1, ((N - 1)) + merge(1, -1, (W) < 0), W
            do jj = 1, ((N - 1)) + merge(1, -1, (W) < 0), W
                do i = ii, (min((ii + W), (N - 1))) - 1
                    do j = jj, (min((jj + W), (N - 1))) - 1
                        B((j) + 1, (i) + 1) = (0.2_c_double * ((((A((j) + 1, (i) + 1) + A(((j - 1)) + 1, (i) + 1)) + &
                        &A(((j + 1)) + 1, (i) + 1)) + A((j) + 1, ((i - 1)) + 1)) + A((j) + 1, ((i + 1)) + 1)))
                    end do
                end do
            end do
        end do
        do ii = 1, ((N - 1)) + merge(1, -1, (W) < 0), W
            do jj = 1, ((N - 1)) + merge(1, -1, (W) < 0), W
                do i = ii, (min((ii + W), (N - 1))) - 1
                    do j = jj, (min((jj + W), (N - 1))) - 1
                        A((j) + 1, (i) + 1) = (0.2_c_double * ((((B((j) + 1, (i) + 1) + B(((j - 1)) + 1, (i) + 1)) + &
                        &B(((j + 1)) + 1, (i) + 1)) + B((j) + 1, ((i - 1)) + 1)) + B((j) + 1, ((i + 1)) + 1)))
                    end do
                end do
            end do
        end do
    end do

end subroutine jacobi_2d_tile_w7_fp64
