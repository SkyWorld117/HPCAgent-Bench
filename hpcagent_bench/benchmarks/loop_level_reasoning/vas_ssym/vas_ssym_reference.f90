! Fortran baseline reference for HPCAgent-Bench kernel vas_ssym, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from vas_ssym_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine vas_ssym_fp64(a, b, ip, LEN_1D, SSYM) bind(C, name="vas_ssym_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t), value, intent(in) :: SSYM
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int64_t), intent(in) :: ip(LEN_1D)
    integer(c_int64_t) :: i

    do i = 0, (npb_floordiv_i(INT(LEN_1D, c_int64_t), INT(SSYM, c_int64_t))) - 1
        a((ip(((i * SSYM)) + 1)) + 1) = b((i) + 1)
    end do
contains

    elemental function npb_floordiv_i(a, b) result(r)
        integer(c_int64_t), intent(in) :: a, b
        integer(c_int64_t) :: r
        r = a / b - merge(1_c_int64_t, 0_c_int64_t, (mod(a, b) /= 0_c_int64_t) .and. ((a < 0_c_int64_t) .neqv. (b < &
        &0_c_int64_t)))
    end function npb_floordiv_i

end subroutine vas_ssym_fp64
