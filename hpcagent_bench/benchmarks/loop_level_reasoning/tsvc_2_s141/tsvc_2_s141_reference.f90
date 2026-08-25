! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_s141, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_s141_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s141_fp64(bb, flat_2d_array, LEN_2D) bind(C, name="tsvc_2_s141_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
    real(c_double), intent(inout) :: flat_2d_array(LEN_2D * LEN_2D)
    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: k
    do i = 0, (LEN_2D) - 1
        k = (npb_floordiv_i(INT(((i + 1) * i), c_int64_t), INT(2, c_int64_t)) + i)
        do j = i, (LEN_2D) - 1
            flat_2d_array((k) + 1) = (flat_2d_array((k) + 1) + bb((i) + 1, (j) + 1))
            k = ((k + j) + 1)
        end do
    end do
contains

    elemental function npb_floordiv_i(a, b) result(r)
        integer(c_int64_t), intent(in) :: a, b
        integer(c_int64_t) :: r
        r = a / b - merge(1_c_int64_t, 0_c_int64_t, (mod(a, b) /= 0_c_int64_t) .and. ((a < 0_c_int64_t) .neqv. (b < &
        &0_c_int64_t)))
    end function npb_floordiv_i

end subroutine tsvc_2_s141_fp64
