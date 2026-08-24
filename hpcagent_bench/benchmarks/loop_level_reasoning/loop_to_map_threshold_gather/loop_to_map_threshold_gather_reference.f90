! Fortran baseline reference for HPCAgent-Bench kernel loop_to_map_threshold_gather, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from loop_to_map_threshold_gather_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine loop_to_map_threshold_gather_fp64(idx, out, w, x, y, LEN_2D) bind(C, &
&name="loop_to_map_threshold_gather_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    integer(c_int64_t), intent(in) :: idx(LEN_2D)
    real(c_double), intent(inout) :: out(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: w(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: x(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: y(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, k

    do i = 0, (LEN_2D) - 1
        do k = 0, (LEN_2D) - 1
            if ((w((k) + 1, (idx((i) + 1)) + 1) > 0.5_c_double)) then
                out((k) + 1, (i) + 1) = (x((k) + 1, (i) + 1) * 2.0_c_double)
            else
                out((k) + 1, (i) + 1) = (y((k) + 1, (i) + 1) + 1.0_c_double)
            end if
        end do
    end do

end subroutine loop_to_map_threshold_gather_fp64
