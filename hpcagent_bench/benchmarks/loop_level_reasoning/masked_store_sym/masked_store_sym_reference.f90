! Fortran baseline reference for HPCAgent-Bench kernel masked_store_sym, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from masked_store_sym_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine masked_store_sym_fp64(a, b, threshold_data, K, LEN_1D) bind(C, name="masked_store_sym_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: K
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    real(c_double), intent(in) :: threshold_data(LEN_1D)
    integer(c_int64_t) :: i

    do i = 0, (LEN_1D) - 1
        if ((threshold_data((i) + 1) > K)) then
            a((i) + 1) = b((i) + 1)
        end if
    end do

end subroutine masked_store_sym_fp64
