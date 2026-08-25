! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s13110, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s13110_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s13110_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s13110_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(in) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(inout) :: bb(2, 2)
    integer(c_int64_t) :: i, j
    real(c_double) :: maxv
    real(c_double) :: xindex
    real(c_double) :: yindex
    real(c_double) :: chksum
    real(c_double) :: tmp
    maxv = aa((0) + 1, (0) + 1)
    xindex = 0
    yindex = 0
    do i = 0, (LEN_2D) - 1
        do j = 0, (LEN_2D) - 1
            if ((aa((j) + 1, (i) + 1) > maxv)) then
                maxv = aa((j) + 1, (i) + 1)
                xindex = i
                yindex = j
            end if
        end do
    end do
    chksum = ((maxv + xindex) + yindex)
    tmp = chksum
    bb((0) + 1, (0) + 1) = chksum

end subroutine tsvc_2_s13110_fp64
