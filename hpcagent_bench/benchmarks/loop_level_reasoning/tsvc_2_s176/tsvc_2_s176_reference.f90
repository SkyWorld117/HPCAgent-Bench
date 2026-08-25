! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s176, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s176_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s176_fp64(a, b, c, LEN_1D) bind(C, name="tsvc_2_s176_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D)
    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: m
    m = npb_floordiv_i(INT(LEN_1D, c_int64_t), INT(2, c_int64_t))
    do j = 0, (npb_floordiv_i(INT(LEN_1D, c_int64_t), INT(2, c_int64_t))) - 1
        do i = 0, (m) - 1
            a((i) + 1) = (a((i) + 1) + (b(((((i + m) - j) - 1)) + 1) * c((j) + 1)))
        end do
    end do
contains

    elemental function npb_floordiv_i(a, b) result(r)
        integer(c_int64_t), intent(in) :: a, b
        integer(c_int64_t) :: r
        r = a / b - merge(1_c_int64_t, 0_c_int64_t, (mod(a, b) /= 0_c_int64_t) .and. ((a < 0_c_int64_t) .neqv. (b < &
        &0_c_int64_t)))
    end function npb_floordiv_i

end subroutine tsvc_2_s176_fp64
