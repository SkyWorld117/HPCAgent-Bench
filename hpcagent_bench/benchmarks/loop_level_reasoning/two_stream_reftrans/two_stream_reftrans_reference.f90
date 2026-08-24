! Fortran baseline reference for HPCAgent-Bench kernel two_stream_reftrans, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from two_stream_reftrans_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine two_stream_reftrans_fp64(g1, g2, od, ref, trans, NG) bind(C, name="two_stream_reftrans_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: NG
    real(c_double), intent(in) :: g1(NG)
    real(c_double), intent(in) :: g2(NG)
    real(c_double), intent(in) :: od(NG)
    real(c_double), intent(inout) :: ref(NG)
    real(c_double), intent(inout) :: trans(NG)
    integer(c_int64_t) :: jg
    real(c_double) :: k
    real(c_double) :: e
    real(c_double) :: e2
    real(c_double) :: rf
    do jg = 0, (NG) - 1
        if ((od((jg) + 1) > 0.001_c_double)) then
            k = SQRT(npb_max2(((g1((jg) + 1) - g2((jg) + 1)) * (g1((jg) + 1) + g2((jg) + 1))), 1e-12_c_double))
            e = EXP(((-(k)) * od((jg) + 1)))
            e2 = (e * e)
            rf = (1.0_c_double / ((k + g1((jg) + 1)) + ((k - g1((jg) + 1)) * e2)))
            ref((jg) + 1) = ((g2((jg) + 1) * (1.0_c_double - e2)) * rf)
            trans((jg) + 1) = (((2.0_c_double * k) * e) * rf)
        else
            ref((jg) + 1) = (g2((jg) + 1) * od((jg) + 1))
            trans((jg) + 1) = (1.0_c_double - (g1((jg) + 1) * od((jg) + 1)))
        end if
        ref((jg) + 1) = npb_max2(0.0_c_double, npb_min2(ref((jg) + 1), 1.0_c_double))
        trans((jg) + 1) = npb_max2(0.0_c_double, npb_min2(trans((jg) + 1), (1.0_c_double - ref((jg) + 1))))
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

end subroutine two_stream_reftrans_fp64
