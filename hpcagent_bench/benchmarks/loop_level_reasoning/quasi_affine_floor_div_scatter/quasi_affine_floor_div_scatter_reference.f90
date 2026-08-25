! Fortran baseline reference for HPCAgent-Bench kernel quasi_affine_floor_div_scatter, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from quasi_affine_floor_div_scatter_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine quasi_affine_floor_div_scatter_fp64(a, b, LEN_1D) bind(C, name="quasi_affine_floor_div_scatter_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(2 * LEN_1D)
    real(c_double), intent(inout) :: b(LEN_1D)
    integer(c_int64_t) :: i

    do i = 0, ((2 * LEN_1D)) - 1
        b((npb_floordiv_i(INT(i, c_int64_t), INT(2, c_int64_t))) + 1) = (b((npb_floordiv_i(INT(i, c_int64_t), INT(2, &
        &c_int64_t))) + 1) + a((i) + 1))
    end do
contains

    elemental function npb_floordiv_i(a, b) result(r)
        integer(c_int64_t), intent(in) :: a, b
        integer(c_int64_t) :: r
        r = a / b - merge(1_c_int64_t, 0_c_int64_t, (mod(a, b) /= 0_c_int64_t) .and. ((a < 0_c_int64_t) .neqv. (b < &
        &0_c_int64_t)))
    end function npb_floordiv_i

end subroutine quasi_affine_floor_div_scatter_fp64
