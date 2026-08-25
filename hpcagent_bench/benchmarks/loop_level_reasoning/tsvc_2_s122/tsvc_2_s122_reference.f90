! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s122, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s122_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s122_fp64(a, b, LEN_1D, n1, n3) bind(C, name="tsvc_2_s122_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t), value, intent(in) :: n1
    integer(c_int64_t), value, intent(in) :: n3
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int64_t) :: i
    integer(c_int64_t) :: j
    integer(c_int64_t) :: k
    j = 1
    k = 0
    do i = (n1 - 1), (LEN_1D) + merge(1, -1, (n3) < 0), n3
        k = (k + j)
        a((i) + 1) = (a((i) + 1) + b(((LEN_1D - k)) + 1))
    end do

end subroutine tsvc_2_s122_fp64
