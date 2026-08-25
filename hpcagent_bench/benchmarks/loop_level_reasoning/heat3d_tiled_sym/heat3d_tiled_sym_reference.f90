! Fortran baseline reference for HPCAgent-Bench kernel heat3d_tiled_sym, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from heat3d_tiled_sym_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine heat3d_tiled_sym_fp64(a, b, LEN_3D, T) bind(C, name="heat3d_tiled_sym_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_3D
    integer(c_int64_t), value, intent(in) :: T
    real(c_double), intent(in) :: a(LEN_3D, LEN_3D, LEN_3D)
    real(c_double), intent(inout) :: b(LEN_3D, LEN_3D, LEN_3D)
    integer(c_int64_t) :: i, ii, j, jj, k, kk

    do kk = 1, (((LEN_3D - 1) - T)) + merge(1, -1, (T) < 0), T
        do jj = 1, (((LEN_3D - 1) - T)) + merge(1, -1, (T) < 0), T
            do ii = 1, (((LEN_3D - 1) - T)) + merge(1, -1, (T) < 0), T
                do k = kk, ((kk + T)) - 1
                    do j = jj, ((jj + T)) - 1
                        do i = ii, ((ii + T)) - 1
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

end subroutine heat3d_tiled_sym_fp64
