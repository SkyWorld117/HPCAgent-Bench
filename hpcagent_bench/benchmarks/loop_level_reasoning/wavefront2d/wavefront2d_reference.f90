! Fortran baseline reference for HPCAgent-Bench kernel wavefront2d, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from wavefront2d_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine wavefront2d_fp64(a, LEN_2D) bind(C, name="wavefront2d_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, j

    do i = 1, (LEN_2D) - 1
        do j = 1, (LEN_2D) - 1
            a((j) + 1, (i) + 1) = (0.25_c_double * (((a((j) + 1, (i) + 1) + a((j) + 1, ((i - 1)) + 1)) + a(((j - 1)) + &
            &1, (i) + 1)) + a(((j - 1)) + 1, ((i - 1)) + 1)))
        end do
    end do

end subroutine wavefront2d_fp64
