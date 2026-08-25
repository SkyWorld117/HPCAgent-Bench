! Fortran baseline reference for HPCAgent-Bench kernel ext_tile_2d_sym, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from ext_tile_2d_sym_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine ext_tile_2d_sym_fp64(a, b, LEN_2D, S) bind(C, name="ext_tile_2d_sym_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    integer(c_int64_t), value, intent(in) :: S
    real(c_double), intent(in) :: a(LEN_2D, LEN_2D)
    real(c_double), intent(inout) :: b(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, j, ti, tj

    do ti = 0, (LEN_2D) + merge(1, -1, (S) < 0), S
        do tj = 0, (LEN_2D) + merge(1, -1, (S) < 0), S
            do i = ti, ((ti + S)) - 1
                do j = tj, ((tj + S)) - 1
                    b((j) + 1, (i) + 1) = (a((j) + 1, (i) + 1) * 2.0_c_double)
                end do
            end do
        end do
    end do

end subroutine ext_tile_2d_sym_fp64
