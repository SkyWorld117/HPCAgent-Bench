! Fortran baseline reference for HPCAgent-Bench kernel thomas_solve, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from thomas_solve_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine thomas_solve_fp64(a, b, c, d, x, LEN_1D) bind(C, name="thomas_solve_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    real(c_double), intent(inout) :: c(LEN_1D)
    real(c_double), intent(inout) :: d(LEN_1D)
    real(c_double), intent(inout) :: x(LEN_1D)
    integer(c_int64_t) :: i
    real(c_double) :: m
    c((0) + 1) = (c((0) + 1) / b((0) + 1))
    d((0) + 1) = (d((0) + 1) / b((0) + 1))
    do i = 1, (LEN_1D) - 1
        m = (b((i) + 1) - (a((i) + 1) * c(((i - 1)) + 1)))
        c((i) + 1) = (c((i) + 1) / m)
        d((i) + 1) = ((d((i) + 1) - (a((i) + 1) * d(((i - 1)) + 1))) / m)
    end do
    x(((LEN_1D - 1)) + 1) = d(((LEN_1D - 1)) + 1)
    do i = (LEN_1D - 2), ((-1)) + 1, (-1)
        x((i) + 1) = (d((i) + 1) - (c((i) + 1) * x(((i + 1)) + 1)))
    end do

end subroutine thomas_solve_fp64
