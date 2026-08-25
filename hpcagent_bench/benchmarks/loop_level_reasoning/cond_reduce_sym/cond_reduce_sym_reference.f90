! Fortran baseline reference for HPCAgent-Bench kernel cond_reduce_sym, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from cond_reduce_sym_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine cond_reduce_sym_fp64(a, out, K, LEN_1D) bind(C, name="cond_reduce_sym_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: K
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: out(1)
    integer(c_int64_t) :: i

    out((0) + 1) = 0.0_c_double
    do i = 0, (LEN_1D) - 1
        if ((a((i) + 1) > K)) then
            out((0) + 1) = (out((0) + 1) + a((i) + 1))
        end if
    end do

end subroutine cond_reduce_sym_fp64
