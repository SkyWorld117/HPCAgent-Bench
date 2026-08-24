! Fortran baseline reference for HPCAgent-Bench kernel reduce_inner_carry, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from reduce_inner_carry_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine reduce_inner_carry_fp64(a, out, LEN_2D) bind(C, name="reduce_inner_carry_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(in) :: a(LEN_2D, LEN_2D)
    real(c_double), intent(inout) :: out(LEN_2D)
    integer(c_int64_t) :: i, j
    real(c_double) :: s
    do i = 0, (LEN_2D) - 1
        s = 0.0_c_double
        do j = 0, (LEN_2D) - 1
            s = (s + a((j) + 1, (i) + 1))
        end do
        out((i) + 1) = s
    end do

end subroutine reduce_inner_carry_fp64
