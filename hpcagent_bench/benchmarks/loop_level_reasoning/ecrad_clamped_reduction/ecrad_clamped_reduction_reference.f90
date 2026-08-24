! Fortran baseline reference for HPCAgent-Bench kernel ecrad_clamped_reduction, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from ecrad_clamped_reduction_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine ecrad_clamped_reduction_fp64(d, out, x, y, LEN_1D) bind(C, name="ecrad_clamped_reduction_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: d(LEN_1D)
    real(c_double), intent(inout) :: out(LEN_1D)
    real(c_double), intent(in) :: x(LEN_1D)
    real(c_double), intent(in) :: y(LEN_1D)
    integer(c_int64_t) :: i
    real(c_double) :: k
    real(c_double) :: e
    do i = 0, (LEN_1D) - 1
        k = SQRT(npb_max2(((x((i) + 1) * x((i) + 1)) + (y((i) + 1) * y((i) + 1))), 1e-12_c_double))
        e = EXP(((-(k)) * d((i) + 1)))
        out((i) + 1) = npb_max2(0.0_c_double, npb_min2(e, 1.0_c_double))
    end do
contains

    elemental function npb_min2(a, b) result(r)
        real(c_double), intent(in) :: a, b
        real(c_double) :: r
        r = merge(a + b, merge(a, b, a < b), (a /= a) .or. (b /= b))
    end function npb_min2

    elemental function npb_max2(a, b) result(r)
        real(c_double), intent(in) :: a, b
        real(c_double) :: r
        r = merge(a + b, merge(a, b, a > b), (a /= a) .or. (b /= b))
    end function npb_max2

end subroutine ecrad_clamped_reduction_fp64
