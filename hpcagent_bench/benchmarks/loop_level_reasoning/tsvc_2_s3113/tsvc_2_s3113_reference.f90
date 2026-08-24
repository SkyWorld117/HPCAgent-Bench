! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s3113, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s3113_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s3113_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s3113_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: b(2)
    integer(c_int64_t) :: i
    real(c_double) :: maxv
    real(c_double) :: av
    maxv = 0.0_c_double
    maxv = ABS(a((0) + 1))
    do i = 0, (LEN_1D) - 1
        av = ABS(a((i) + 1))
        if ((av > maxv)) then
            maxv = av
        end if
    end do
    b((0) + 1) = maxv

end subroutine tsvc_2_s3113_fp64
