! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s318, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s318_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc) bind(C, name="tsvc_2_s318_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t), value, intent(in) :: inc
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: result(1)
    integer(c_int64_t) :: i
    integer(c_int64_t) :: k
    real(c_double) :: index
    real(c_double) :: maxv
    real(c_double) :: v
    k = 0
    index = 0
    maxv = ABS(a((0) + 1))
    k = (k + inc)
    do i = 1, (LEN_1D) - 1
        v = ABS(a((k) + 1))
        if ((v > maxv)) then
            index = i
            maxv = v
        end if
        k = (k + inc)
    end do
    result((0) + 1) = (maxv + index)

end subroutine tsvc_2_s318_fp64
