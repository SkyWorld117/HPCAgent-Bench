! Fortran baseline reference for HPCAgent-Bench kernel jacobi_2d_tile_4lvl_silly, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from jacobi_2d_tile_4lvl_silly_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine jacobi_2d_tile_4lvl_silly_fp64(A, B, N, TSTEPS) bind(C, name="jacobi_2d_tile_4lvl_silly_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: N
    integer(c_int64_t), value, intent(in) :: TSTEPS
    real(c_double), intent(inout) :: A(N, N)
    real(c_double), intent(inout) :: B(N, N)
    integer(c_int64_t) :: i, i1, i2, i3, i4, j, j1, j2, j3, j4, si0, si1, t
    integer(c_int64_t) :: W1
    integer(c_int64_t) :: W2
    integer(c_int64_t) :: W3
    integer(c_int64_t) :: W4
    W1 = 13
    W2 = 7
    W3 = 19
    W4 = 3
    do t = 0, (TSTEPS) - 1
        do i1 = 1, ((N - 1)) + merge(1, -1, (W1) < 0), W1
            do j1 = 1, ((N - 1)) + merge(1, -1, (W1) < 0), W1
                do i2 = i1, (min((i1 + W1), (N - 1))) + merge(1, -1, (W2) < 0), W2
                    do j2 = j1, (min((j1 + W1), (N - 1))) + merge(1, -1, (W2) < 0), W2
                        do i3 = i2, (min((i2 + W2), (N - 1))) + merge(1, -1, (W3) < 0), W3
                            do j3 = j2, (min((j2 + W2), (N - 1))) + merge(1, -1, (W3) < 0), W3
                                do i4 = i3, (min((i3 + W3), (N - 1))) + merge(1, -1, (W4) < 0), W4
                                    do j4 = j3, (min((j3 + W3), (N - 1))) + merge(1, -1, (W4) < 0), W4
                                        do i = i4, (min((i4 + W4), (N - 1))) - 1
                                            do j = j4, (min((j4 + W4), (N - 1))) - 1
                                                B((j) + 1, (i) + 1) = (0.2_c_double * ((((A((j) + 1, (i) + 1) + A(((j &
                                                &- 1)) + 1, (i) + 1)) + A(((j + 1)) + 1, (i) + 1)) + A((j) + 1, ((i - &
                                                &1)) + 1)) + A((j) + 1, ((i + 1)) + 1)))
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do
                    end do
                end do
            end do
        end do
        do si0 = 1, ((N - 1)) - 1
            do si1 = 1, ((N - 1)) - 1
                A((si1) + 1, (si0) + 1) = B((si1) + 1, (si0) + 1)
            end do
        end do
    end do

end subroutine jacobi_2d_tile_4lvl_silly_fp64
