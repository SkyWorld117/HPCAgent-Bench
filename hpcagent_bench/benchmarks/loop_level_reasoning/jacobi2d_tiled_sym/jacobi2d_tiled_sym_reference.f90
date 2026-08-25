! Fortran baseline reference for HPCAgent-Bench kernel jacobi2d_tiled_sym, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from jacobi2d_tiled_sym_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine jacobi2d_tiled_sym_fp64(a, b, LEN_2D, T) bind(C, name="jacobi2d_tiled_sym_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    integer(c_int64_t), value, intent(in) :: T
    real(c_double), intent(in) :: a(LEN_2D, LEN_2D)
    real(c_double), intent(inout) :: b(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, ii, j, jj

    do ii = 1, (((LEN_2D - 1) - T)) + merge(1, -1, (T) < 0), T
        do jj = 1, (((LEN_2D - 1) - T)) + merge(1, -1, (T) < 0), T
            do i = ii, ((ii + T)) - 1
                do j = jj, ((jj + T)) - 1
                    b((j) + 1, (i) + 1) = (0.2_c_double * ((((a((j) + 1, (i) + 1) + a((j) + 1, ((i - 1)) + 1)) + a((j) &
                    &+ 1, ((i + 1)) + 1)) + a(((j - 1)) + 1, (i) + 1)) + a(((j + 1)) + 1, (i) + 1)))
                end do
            end do
        end do
    end do

end subroutine jacobi2d_tiled_sym_fp64
