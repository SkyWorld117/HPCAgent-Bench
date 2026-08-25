! Fortran baseline reference for HPCAgent-Bench kernel wavefront_2d, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from wavefront_2d_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine wavefront_2d_fp64(aa, N) bind(C, name="wavefront_2d_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: N
    real(c_double), intent(inout) :: aa(N, N)
    integer(c_int64_t) :: i, j

    do i = 1, (N) - 1
        do j = 1, (N) - 1
            aa((j) + 1, (i) + 1) = ((aa(((j - 1)) + 1, (i) + 1) + aa((j) + 1, ((i - 1)) + 1)) / 1.9_c_double)
        end do
    end do

end subroutine wavefront_2d_fp64
