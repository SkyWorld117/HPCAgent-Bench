! Fortran baseline reference for HPCAgent-Bench kernel heat3d_tiled_const, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from heat3d_tiled_const_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine heat3d_tiled_const_fp64(a, b, LEN_3D) bind(C, name="heat3d_tiled_const_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_3D
    real(c_double), intent(in) :: a(LEN_3D, LEN_3D, LEN_3D)
    real(c_double), intent(inout) :: b(LEN_3D, LEN_3D, LEN_3D)
    integer(c_int64_t) :: i, ii, j, jj, k, kk

    do kk = 1, (((LEN_3D - 1) - 8)) - 1, 8
        do jj = 1, (((LEN_3D - 1) - 8)) - 1, 8
            do ii = 1, (((LEN_3D - 1) - 8)) - 1, 8
                do k = kk, ((kk + 8)) - 1
                    do j = jj, ((jj + 8)) - 1
                        do i = ii, ((ii + 8)) - 1
                            b((i) + 1, (j) + 1, (k) + 1) = ((((0.125_c_double * ((a((i) + 1, (j) + 1, ((k + 1)) + 1) - &
                            &(2.0_c_double * a((i) + 1, (j) + 1, (k) + 1))) + a((i) + 1, (j) + 1, ((k - 1)) + 1))) + &
                            &(0.125_c_double * ((a((i) + 1, ((j + 1)) + 1, (k) + 1) - (2.0_c_double * a((i) + 1, (j) + &
                            &1, (k) + 1))) + a((i) + 1, ((j - 1)) + 1, (k) + 1)))) + (0.125_c_double * ((a(((i + 1)) + &
                            &1, (j) + 1, (k) + 1) - (2.0_c_double * a((i) + 1, (j) + 1, (k) + 1))) + a(((i - 1)) + 1, &
                            &(j) + 1, (k) + 1)))) + a((i) + 1, (j) + 1, (k) + 1))
                        end do
                    end do
                end do
            end do
        end do
    end do

end subroutine heat3d_tiled_const_fp64
