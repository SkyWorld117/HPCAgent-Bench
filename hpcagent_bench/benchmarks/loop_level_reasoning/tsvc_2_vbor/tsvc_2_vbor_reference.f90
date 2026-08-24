! Fortran baseline reference for HPCAgent-Bench kernel tsvc_2_vbor, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from tsvc_2_vbor_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_vbor_fp64(a, b, c, d, e, x, LEN_2D) bind(C, name="tsvc_2_vbor_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(in) :: a(LEN_2D)
    real(c_double), intent(in) :: b(LEN_2D)
    real(c_double), intent(in) :: c(LEN_2D)
    real(c_double), intent(in) :: d(LEN_2D)
    real(c_double), intent(in) :: e(LEN_2D)
    real(c_double), intent(inout) :: x(LEN_2D)
    integer(c_int64_t) :: i
    real(c_double) :: a1
    real(c_double) :: b1
    real(c_double) :: c1
    real(c_double) :: d1
    real(c_double) :: e1
    real(c_double) :: f1
    do i = 0, (LEN_2D) - 1
        a1 = a((i) + 1)
        b1 = b((i) + 1)
        c1 = c((i) + 1)
        d1 = d((i) + 1)
        e1 = e((i) + 1)
        f1 = a((i) + 1)
        a1 = (((((((((((a1 * b1) * c1) + ((a1 * b1) * d1)) + ((a1 * b1) * e1)) + ((a1 * b1) * f1)) + ((a1 * c1) * d1)) &
        &+ ((a1 * c1) * e1)) + ((a1 * c1) * f1)) + ((a1 * d1) * e1)) + ((a1 * d1) * f1)) + ((a1 * e1) * f1))
        b1 = (((((((b1 * c1) * d1) + ((b1 * c1) * e1)) + ((b1 * c1) * f1)) + ((b1 * d1) * e1)) + ((b1 * d1) * f1)) + &
        &((b1 * e1) * f1))
        c1 = ((((c1 * d1) * e1) + ((c1 * d1) * f1)) + ((c1 * e1) * f1))
        d1 = ((d1 * e1) * f1)
        x((i) + 1) = (((a1 * b1) * c1) * d1)
    end do

end subroutine tsvc_2_vbor_fp64
