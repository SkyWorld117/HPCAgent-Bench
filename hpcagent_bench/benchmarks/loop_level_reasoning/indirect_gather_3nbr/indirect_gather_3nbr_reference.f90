! Fortran baseline reference for HPCAgent-Bench kernel indirect_gather_3nbr, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from indirect_gather_3nbr_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine indirect_gather_3nbr_fp64(field, idx, out, w, N) bind(C, name="indirect_gather_3nbr_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: N
    real(c_double), intent(in) :: field(N)
    integer(c_int64_t), intent(in) :: idx(3, N)
    real(c_double), intent(inout) :: out(N)
    real(c_double), intent(in) :: w(3, N)
    integer(c_int64_t) :: jc

    do jc = 0, (N) - 1
        out((jc) + 1) = (((w((0) + 1, (jc) + 1) * field((idx((0) + 1, (jc) + 1)) + 1)) + (w((1) + 1, (jc) + 1) * &
        &field((idx((1) + 1, (jc) + 1)) + 1))) + (w((2) + 1, (jc) + 1) * field((idx((2) + 1, (jc) + 1)) + 1)))
    end do

end subroutine indirect_gather_3nbr_fp64
