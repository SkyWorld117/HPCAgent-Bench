! Fortran baseline reference for HPCAgent-Bench kernel twin_reduction_shared_stencil, emitted by HPCAgent-Bench's NumpyToX
! Fortran translator (numpyto_fortran) from the numpy reference. The v2 C-ABI carries no timer.
! Not the scoring oracle -- the numpy reference remains the correctness oracle.

! hpcagent_bench-autogen -- generated from twin_reduction_shared_stencil_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine twin_reduction_shared_stencil_fp64(div_mass, div_theta, gfac, idx, mass_fl, theta_fl, N) bind(C, &
&name="twin_reduction_shared_stencil_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: N
    real(c_double), intent(inout) :: div_mass(N)
    real(c_double), intent(inout) :: div_theta(N)
    real(c_double), intent(in) :: gfac(3, N)
    integer(c_int64_t), intent(in) :: idx(3, N)
    real(c_double), intent(in) :: mass_fl(N)
    real(c_double), intent(in) :: theta_fl(N)
    integer(c_int64_t) :: jc

    do jc = 0, (N) - 1
        div_mass((jc) + 1) = (((mass_fl((idx((0) + 1, (jc) + 1)) + 1) * gfac((0) + 1, (jc) + 1)) + (mass_fl((idx((1) + &
        &1, (jc) + 1)) + 1) * gfac((1) + 1, (jc) + 1))) + (mass_fl((idx((2) + 1, (jc) + 1)) + 1) * gfac((2) + 1, (jc) &
        &+ 1)))
        div_theta((jc) + 1) = (((theta_fl((idx((0) + 1, (jc) + 1)) + 1) * gfac((0) + 1, (jc) + 1)) + &
        &(theta_fl((idx((1) + 1, (jc) + 1)) + 1) * gfac((1) + 1, (jc) + 1))) + (theta_fl((idx((2) + 1, (jc) + 1)) + 1) &
        &* gfac((2) + 1, (jc) + 1)))
    end do

end subroutine twin_reduction_shared_stencil_fp64
