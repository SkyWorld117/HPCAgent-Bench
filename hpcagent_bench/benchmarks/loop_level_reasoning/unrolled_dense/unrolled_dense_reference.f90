! Fortran baseline reference for HPCAgent-Bench kernel unrolled_dense, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from unrolled_dense_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine unrolled_dense_fp64(a, b, N, alpha) bind(C, name="unrolled_dense_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: N
    real(c_double), intent(inout) :: a(N)
    real(c_double), intent(in) :: b(N)
    real(c_double), value, intent(in) :: alpha
    integer(c_int64_t) :: i

    do i = 0, ((N - 3)) - 1, 4
        a((i) + 1) = (a((i) + 1) + (alpha * b((i) + 1)))
        a(((i + 1)) + 1) = (a(((i + 1)) + 1) + (alpha * b(((i + 1)) + 1)))
        a(((i + 2)) + 1) = (a(((i + 2)) + 1) + (alpha * b(((i + 2)) + 1)))
        a(((i + 3)) + 1) = (a(((i + 3)) + 1) + (alpha * b(((i + 3)) + 1)))
    end do

end subroutine unrolled_dense_fp64
