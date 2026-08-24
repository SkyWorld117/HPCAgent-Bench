! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s114, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s114_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s114_fp64(aa, bb, LEN_2D, VLEN) bind(C, name="tsvc_2_s114_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    integer(c_int64_t), value, intent(in) :: VLEN
    real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, j

    do i = 0, (npb_floordiv_i(INT(LEN_2D, c_int64_t), INT(VLEN, c_int64_t))) - 1
        do j = 0, ((i * VLEN)) - 1
            aa((j) + 1, (i) + 1) = (aa((i) + 1, (j) + 1) + bb((j) + 1, (i) + 1))
        end do
    end do
contains

    elemental function npb_floordiv_i(a, b) result(r)
        integer(c_int64_t), intent(in) :: a, b
        integer(c_int64_t) :: r
        r = a / b - merge(1_c_int64_t, 0_c_int64_t, (mod(a, b) /= 0_c_int64_t) .and. ((a < 0_c_int64_t) .neqv. (b < &
        &0_c_int64_t)))
    end function npb_floordiv_i

end subroutine tsvc_2_s114_fp64
