! Fortran baseline reference for HPCAgent-Bench kernel vertical_flux_prefix_scan, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from vertical_flux_prefix_scan_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine vertical_flux_prefix_scan_fp64(fall, flux, K, N) bind(C, name="vertical_flux_prefix_scan_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: K
    integer(c_int64_t), value, intent(in) :: N
    real(c_double), intent(in) :: fall(K, N)
    real(c_double), intent(inout) :: flux(K, N)
    integer(c_int64_t) :: i, kk

    do i = 0, (N) - 1
        flux((0) + 1, (i) + 1) = fall((0) + 1, (i) + 1)
        do kk = 1, (K) - 1
            flux((kk) + 1, (i) + 1) = ((flux(((kk - 1)) + 1, (i) + 1) * 0.9_c_double) + fall((kk) + 1, (i) + 1))
        end do
    end do

end subroutine vertical_flux_prefix_scan_fp64
