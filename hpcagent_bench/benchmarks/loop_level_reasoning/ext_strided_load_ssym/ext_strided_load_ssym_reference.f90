! Fortran baseline reference for HPCAgent-Bench kernel ext_strided_load_ssym, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from ext_strided_load_ssym_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine ext_strided_load_ssym_fp64(dst, src, LEN_1D, SSYM, scale) bind(C, name="ext_strided_load_ssym_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t), value, intent(in) :: SSYM
    real(c_double), intent(inout) :: dst(LEN_1D)
    real(c_double), intent(in) :: src(SSYM * LEN_1D)
    real(c_double), value, intent(in) :: scale
    integer(c_int64_t) :: i

    do i = 0, (LEN_1D) - 1
        dst((i) + 1) = (src(((i * SSYM)) + 1) * scale)
    end do

end subroutine ext_strided_load_ssym_fp64
