"""Vectorized numpy port of ICON mo_velocity_advection.velocity_tendencies.

Same math as velocity_tendencies_numpy.py. Every "for jk in range(...)" level loop is dropped by
gathering/broadcasting over the whole level axis at once; the small per-neighbour loops (3, 4 or 6
taps -- cell/edge/vertex connectivity is fixed-degree) are kept as tap loops over full-array slices,
mirroring the gather/tap-loop rules for an unstructured-grid stencil.
"""
import numpy as np


def velocity_tendencies(
    p_patch_cells_area,
    p_patch_cells_neighbor_idx,
    p_patch_cells_neighbor_blk,
    p_patch_cells_edge_idx,
    p_patch_cells_edge_blk,
    p_patch_cells_start_index,
    p_patch_cells_end_index,
    p_patch_cells_start_block,
    p_patch_cells_end_block,
    p_patch_cells_decomp_info_owner_mask,
    p_patch_edges_cell_idx,
    p_patch_edges_cell_blk,
    p_patch_edges_vertex_idx,
    p_patch_edges_vertex_blk,
    p_patch_edges_quad_idx,
    p_patch_edges_quad_blk,
    p_patch_edges_tangent_orientation,
    p_patch_edges_inv_primal_edge_length,
    p_patch_edges_inv_dual_edge_length,
    p_patch_edges_area_edge,
    p_patch_edges_f_e,
    p_patch_edges_fn_e,
    p_patch_edges_ft_e,
    p_patch_edges_start_index,
    p_patch_edges_end_index,
    p_patch_edges_start_block,
    p_patch_edges_end_block,
    p_patch_verts_cell_idx,
    p_patch_verts_cell_blk,
    p_patch_verts_edge_idx,
    p_patch_verts_edge_blk,
    p_patch_verts_start_index,
    p_patch_verts_end_index,
    p_patch_verts_start_block,
    p_patch_verts_end_block,
    p_int_c_lin_e,
    p_int_e_bln_c_s,
    p_int_cells_aw_verts,
    p_int_rbf_vec_coeff_e,
    p_int_geofac_grdiv,
    p_int_geofac_rot,
    p_int_geofac_n2s,
    p_prog_w,
    p_prog_vn,
    p_diag_vn_ie_ubc,
    p_diag_vt,
    p_diag_vn_ie,
    p_diag_w_concorr_c,
    p_diag_ddt_vn_apc_pc,
    p_diag_ddt_vn_cor_pc,
    p_diag_ddt_w_adv_pc,
    p_diag_max_vcfl_dyn,
    p_metrics_ddxn_z_full,
    p_metrics_ddxt_z_full,
    p_metrics_ddqz_z_full_e,
    p_metrics_ddqz_z_half,
    p_metrics_wgtfac_c,
    p_metrics_wgtfac_e,
    p_metrics_wgtfacq_e,
    p_metrics_coeff_gradekin,
    p_metrics_coeff1_dwdz,
    p_metrics_coeff2_dwdz,
    p_metrics_deepatmo_gradh_mc,
    p_metrics_deepatmo_invr_mc,
    p_metrics_deepatmo_gradh_ifc,
    p_metrics_deepatmo_invr_ifc,
    z_w_concorr_me,
    z_kin_hor_e,
    z_vt_ie,
    ntnd,
    istep,
    lvn_only,
    ldeepatmo,
    lextra_diffu,
    l_vert_nested,
    ddt_vn_cor_associated,
    dtime,
    dt_linintp_ubc,
    nrdmax_jg,
    nflatlev_jg,
    nproma,
    nlev,
    nlevp1,
    nblks_c,
    nblks_e,
    nblks_v,
):

    t = ntnd - 1  # 1-based tendency slot -> 0-based
    if nrdmax_jg is None:
        nrdmax_jg = nlev
    nf = nflatlev_jg  # 1-based first flat level

    if lextra_diffu:
        cfl_w_limit = 0.65 / dtime
        scalfac_exdiff = 0.05 / (dtime * (0.85 - cfl_w_limit * dtime))
    else:
        cfl_w_limit = 0.85 / dtime
        scalfac_exdiff = 0.0

    vn = p_prog_vn  # (nproma, nlev,   nblks_e)
    w = p_prog_w  # (nproma, nlevp1, nblks_c)

    def gat(A, idx, blk, n, lvl=slice(None)):
        """A[idx[:,:,n]-1, lvl, blk[:,:,n]-1] with the level axis moved back to position 1.

        idx/blk carry no level dependence, so the whole level range gathers in one advanced-index
        call instead of one call per jk.
        """
        i = idx[:, :, n] - 1
        b = blk[:, :, n] - 1
        return np.moveaxis(A[i, lvl, b], -1, 1)

    # ===== z_w_v = cells2verts_scalar_ri(w, cells_aw_verts) (6 cells/vertex) ==
    vci = p_patch_verts_cell_idx  # (nproma, nblks_v, 6)
    vcb = p_patch_verts_cell_blk
    awv = p_int_cells_aw_verts  # (nproma, 6, nblks_v)
    z_w_v = np.zeros((nproma, nlevp1, nblks_v), dtype=w.dtype)
    if not lvn_only:
        for n in range(6):
            z_w_v += awv[:, n, None, :] * gat(w, vci, vcb, n)

    # ===== zeta = rot_vertex_ri(vn, geofac_rot) (6 edges/vertex) =============
    vei = p_patch_verts_edge_idx  # (nproma, nblks_v, 6)
    veb = p_patch_verts_edge_blk
    grot = p_int_geofac_rot  # (nproma, 6, nblks_v)
    zeta = np.zeros((nproma, nlev, nblks_v), dtype=vn.dtype)
    for n in range(6):
        zeta += gat(vn, vei, veb, n) * grot[:, n, None, :]

    # ===== istep == 1 edge block ===========================================
    vt = p_diag_vt  # (nproma, nlev,   nblks_e)
    vn_ie = p_diag_vn_ie  # (nproma, nlevp1, nblks_e)
    vn_ie_ubc = p_diag_vn_ie_ubc  # (nproma, 2,      nblks_e)
    rbf = p_int_rbf_vec_coeff_e  # (4, nproma, nblks_e)
    qi = p_patch_edges_quad_idx  # (nproma, nblks_e, 4)
    qb = p_patch_edges_quad_blk
    wgtfac_e = p_metrics_wgtfac_e  # (nproma, nlevp1, nblks_e)
    wgtfacq_e = p_metrics_wgtfacq_e  # (nproma, 3, nblks_e)
    ddxn = p_metrics_ddxn_z_full  # (nproma, nlev, nblks_e)
    ddxt = p_metrics_ddxt_z_full

    if istep == 1:
        vt[...] = 0.0
        for n in range(4):
            vt += rbf[n, :, None, :] * gat(vn, qi, qb, n)

        rest = slice(1, nlev)
        prev = slice(0, nlev - 1)
        we = wgtfac_e[:, rest, :]
        vn_ie[:, rest, :] = we * vn[:, rest, :] + (1.0 - we) * vn[:, prev, :]
        z_kin_hor_e[:, rest, :] = 0.5 * (vn[:, rest, :]**2 + vt[:, rest, :]**2)
        if not lvn_only:
            z_vt_ie[:, rest, :] = we * vt[:, rest, :] + (1.0 - we) * vt[:, prev, :]

        flat = slice(nf - 1, nlev)
        z_w_concorr_me[:, flat, :] = vn[:, flat, :] * ddxn[:, flat, :] + vt[:, flat, :] * ddxt[:, flat, :]

        if not l_vert_nested:
            vn_ie[:, 0, :] = vn[:, 0, :]
        else:
            vn_ie[:, 0, :] = vn_ie_ubc[:, 0, :] + dt_linintp_ubc * vn_ie_ubc[:, 1, :]
        z_vt_ie[:, 0, :] = vt[:, 0, :]
        z_kin_hor_e[:, 0, :] = 0.5 * (vn[:, 0, :]**2 + vt[:, 0, :]**2)
        vn_ie[:, nlevp1 - 1, :] = (wgtfacq_e[:, 0, :] * vn[:, nlev - 1, :] + wgtfacq_e[:, 1, :] * vn[:, nlev - 2, :] +
                                   wgtfacq_e[:, 2, :] * vn[:, nlev - 3, :])

    # ===== z_v_grad_w (edges, lvn_only=.false.) ============================
    eci = p_patch_edges_cell_idx  # (nproma, nblks_e, 2)
    ecb = p_patch_edges_cell_blk
    evi = p_patch_edges_vertex_idx  # (nproma, nblks_e, 4)
    evb = p_patch_edges_vertex_blk
    inv_dual = p_patch_edges_inv_dual_edge_length  # (nproma, nblks_e)
    inv_prim = p_patch_edges_inv_primal_edge_length
    tang = p_patch_edges_tangent_orientation
    fn_e = p_patch_edges_fn_e
    ft_e = p_patch_edges_ft_e
    gradh_ifc = p_metrics_deepatmo_gradh_ifc  # (nlevp1,)
    invr_ifc = p_metrics_deepatmo_invr_ifc
    z_v_grad_w = np.zeros((nproma, nlev, nblks_e), dtype=vn_ie.dtype)
    if not lvn_only:
        top = slice(0, nlev)
        vn_ie_top = vn_ie[:, top, :]
        z_vt_ie_top = z_vt_ie[:, top, :]
        z_v_grad_w = (vn_ie_top * inv_dual[:, None, :] * (gat(w, eci, ecb, 0, top) - gat(w, eci, ecb, 1, top)) +
                      z_vt_ie_top * inv_prim[:, None, :] * tang[:, None, :] *
                      (gat(z_w_v, evi, evb, 0, top) - gat(z_w_v, evi, evb, 1, top)))
        if ldeepatmo:
            z_v_grad_w = (z_v_grad_w * gradh_ifc[None, :nlev, None] + vn_ie_top *
                          (vn_ie_top * invr_ifc[None, :nlev, None] - ft_e[:, None, :]) + z_vt_ie_top *
                          (z_vt_ie_top * invr_ifc[None, :nlev, None] + fn_e[:, None, :]))

    # ===== cell block: z_ekinh, w_concorr_c, z_w_con_c(_full), ddt_w_adv ====
    cei = p_patch_cells_edge_idx  # (nproma, nblks_c, 3)
    ceb = p_patch_cells_edge_blk
    ebln = p_int_e_bln_c_s  # (nproma, 3, nblks_c)
    wgtfac_c = p_metrics_wgtfac_c  # (nproma, nlevp1, nblks_c)
    w_concorr_c = p_diag_w_concorr_c  # (nproma, nlev, nblks_c)
    coeff1 = p_metrics_coeff1_dwdz  # (nproma, nlev, nblks_c)
    coeff2 = p_metrics_coeff2_dwdz
    ddt_w_adv = p_diag_ddt_w_adv_pc  # (nproma, nlevp1, nblks_c, 3)
    ddqz_half = p_metrics_ddqz_z_half  # (nproma, nlevp1, nblks_c)
    nbi = p_patch_cells_neighbor_idx  # (nproma, nblks_c, 3)
    nbb = p_patch_cells_neighbor_blk
    geofac_n2s = p_int_geofac_n2s  # (nproma, 4, nblks_c)
    area_c = p_patch_cells_area  # (nproma, nblks_c)
    owner = p_patch_cells_decomp_info_owner_mask != 0  # (nproma, nblks_c) bool

    z_ekinh = np.zeros((nproma, nlev, nblks_c), dtype=z_kin_hor_e.dtype)
    for n in range(3):
        z_ekinh += ebln[:, n, None, :] * gat(z_kin_hor_e, cei, ceb, n)

    if istep == 1:
        z_w_concorr_mc = np.zeros((nproma, nlev, nblks_c), dtype=z_w_concorr_me.dtype)
        flat0 = slice(max(nf - 1, 0), nlev)
        for n in range(3):
            z_w_concorr_mc[:, flat0, :] += ebln[:, n, None, :] * gat(z_w_concorr_me, cei, ceb, n, flat0)
        flat1 = slice(nf, nlev)
        wc = wgtfac_c[:, flat1, :]
        w_concorr_c[:, flat1, :] = wc * z_w_concorr_mc[:, flat1, :] + (1.0 - wc) * z_w_concorr_mc[:, nf - 1:nlev - 1, :]

    z_w_con_c = np.zeros((nproma, nlevp1, nblks_c), dtype=w.dtype)
    z_w_con_c[:, :nlev, :] = w[:, :nlev, :]
    z_w_con_c[:, nlevp1 - 1, :] = 0.0
    z_w_con_c[:, nf:nlev, :] -= w_concorr_c[:, nf:nlev, :]

    vcflmax = np.zeros(nblks_c, dtype=z_w_con_c.dtype)
    cfl_clip = np.zeros((nproma, nlevp1, nblks_c), dtype=np.bool_)
    jk0_lo = max(3, nrdmax_jg - 2) - 1  # 0-based; jk1 range was 1-based inclusive
    jk0_hi = nlev - 4  # inclusive upper (jk1 upper nlev-3, jk0 = jk1-1)
    if jk0_hi >= jk0_lo:
        band = slice(jk0_lo, jk0_hi + 1)
        h = ddqz_half[:, band, :]
        zc = z_w_con_c[:, band, :]
        clip = np.abs(zc) > cfl_w_limit * h  # clip <=> |vcfl| > 0.85
        vcfl = zc * dtime / h
        cfl_clip[:, band, :] = clip
        # abs(vcfl) masked to -inf outside the clip: an unclipped cell must never
        # move vcflmax (the Fortran only updates it inside the clip branch), and
        # plain max is reassociation-safe (order never changes a max's result).
        abs_vcfl_masked = np.where(clip, np.abs(vcfl), -np.inf)
        vcflmax = np.maximum(vcflmax, abs_vcfl_masked.max(axis=(0, 1)))
        lo_clip = clip & (vcfl < -0.85)
        hi_clip = clip & (vcfl > 0.85)
        z_w_con_c[:, band, :] = np.where(lo_clip, -0.85 * h / dtime, np.where(hi_clip, 0.85 * h / dtime, zc))

    z_w_con_c_full = 0.5 * (z_w_con_c[:, :nlev, :] + z_w_con_c[:, 1:nlev + 1, :])

    p_diag_max_vcfl_dyn[0] = max(float(p_diag_max_vcfl_dyn[0]), float(vcflmax.max()))

    if not lvn_only:
        rest = slice(1, nlev)
        prev = slice(0, nlev - 1)
        nxt = slice(2, nlev + 1)
        ddt_w_adv[:, rest, :, t] = -z_w_con_c[:, rest, :] * (w[:, prev, :] * coeff1[:, rest, :] -
                                                              w[:, nxt, :] * coeff2[:, rest, :] + w[:, rest, :] *
                                                              (coeff2[:, rest, :] - coeff1[:, rest, :]))
        for n in range(3):
            ddt_w_adv[:, rest, :, t] += ebln[:, n, None, :] * gat(z_v_grad_w, cei, ceb, n, rest)

        if lextra_diffu and jk0_hi >= jk0_lo:
            band = slice(jk0_lo, jk0_hi + 1)
            mask = cfl_clip[:, band, :] & owner[:, None, :]
            h = ddqz_half[:, band, :]
            zc = z_w_con_c[:, band, :]
            difcoef_c = scalfac_exdiff * np.minimum(0.85 - cfl_w_limit * dtime,
                                                     np.abs(zc) * dtime / h - cfl_w_limit * dtime)
            lap = (w[:, band, :] * geofac_n2s[:, None, 0, :] + gat(w, nbi, nbb, 0, band) * geofac_n2s[:, None, 1, :] +
                   gat(w, nbi, nbb, 1, band) * geofac_n2s[:, None, 2, :] +
                   gat(w, nbi, nbb, 2, band) * geofac_n2s[:, None, 3, :])
            ddt_w_adv[:, band, :, t] += np.where(mask, difcoef_c * area_c[:, None, :] * lap, 0.0)

    # levelmask(jk) = ANY over the cell blocks AND cells (full refinement range).
    levelmask = cfl_clip[:, :nlev, :].any(axis=(0, 2))  # (nlev,)

    # ===== edge block: ddt_vn_apc_pc / ddt_vn_cor_pc =======================
    cgk = p_metrics_coeff_gradekin  # (nproma, 2, nblks_e)
    c_lin_e = p_int_c_lin_e  # (nproma, 2, nblks_e)
    f_e = p_patch_edges_f_e  # (nproma, nblks_e)
    ddqz_e = p_metrics_ddqz_z_full_e  # (nproma, nlev, nblks_e)
    ddt_vn_apc = p_diag_ddt_vn_apc_pc  # (nproma, nlev, nblks_e, 3)
    ddt_vn_cor = p_diag_ddt_vn_cor_pc
    geofac_grdiv = p_int_geofac_grdiv  # (nproma, 5, nblks_e)
    area_edge = p_patch_edges_area_edge
    gradh_mc = p_metrics_deepatmo_gradh_mc  # (nlev,)
    invr_mc = p_metrics_deepatmo_invr_mc

    ekc1 = gat(z_ekinh, eci, ecb, 0)
    ekc2 = gat(z_ekinh, eci, ecb, 1)
    zv1 = gat(zeta, evi, evb, 0)
    zv2 = gat(zeta, evi, evb, 1)
    wcf1 = gat(z_w_con_c_full, eci, ecb, 0)
    wcf2 = gat(z_w_con_c_full, eci, ecb, 1)
    clin = c_lin_e[:, None, 0, :] * wcf1 + c_lin_e[:, None, 1, :] * wcf2
    grad_ekin = (z_kin_hor_e * (cgk[:, None, 0, :] - cgk[:, None, 1, :]) + cgk[:, None, 1, :] * ekc2 -
                 cgk[:, None, 0, :] * ekc1)
    dvn = (vn_ie[:, :nlev, :] - vn_ie[:, 1:nlev + 1, :]) / ddqz_e
    if not ldeepatmo:
        ddt_vn_apc[:, :, :, t] = -(grad_ekin + vt * (f_e[:, None, :] + 0.5 * (zv1 + zv2)) + clin * dvn)
        if ddt_vn_cor_associated:
            ddt_vn_cor[:, :, :, t] = -vt * f_e[:, None, :]
    else:
        gmc = gradh_mc[None, :, None]
        rmc = invr_mc[None, :, None]
        ddt_vn_apc[:, :, :, t] = -(grad_ekin * gmc + vt * (f_e[:, None, :] + 0.5 * (zv1 + zv2) * gmc) + clin *
                                   (dvn + vn * rmc - ft_e[:, None, :]))
        if ddt_vn_cor_associated:
            ddt_vn_cor[:, :, :, t] = -(vt * f_e[:, None, :] + clin * (-ft_e[:, None, :]))

    # Background diffusion on the vn-tendency at CFL-flagged levels.
    if lextra_diffu:
        jk0_hi_vn = nlev - 5  # inclusive (jk1 upper nlev-4, jk0 = jk1-1)
        if jk0_hi_vn >= jk0_lo:
            band = slice(jk0_lo, jk0_hi_vn + 1)
            band_next = slice(jk0_lo + 1, jk0_hi_vn + 2)
            lvl_active = levelmask[band] | levelmask[band_next]
            w_con_e = (c_lin_e[:, None, 0, :] * gat(z_w_con_c_full, eci, ecb, 0, band) +
                       c_lin_e[:, None, 1, :] * gat(z_w_con_c_full, eci, ecb, 1, band))
            ddqz_band = ddqz_e[:, band, :]
            clip_e = np.abs(w_con_e) > cfl_w_limit * ddqz_band
            difcoef_e = scalfac_exdiff * np.minimum(0.85 - cfl_w_limit * dtime,
                                                     np.abs(w_con_e) * dtime / ddqz_band - cfl_w_limit * dtime)
            grad = (geofac_grdiv[:, None, 0, :] * vn[:, band, :] +
                    geofac_grdiv[:, None, 1, :] * gat(vn, qi, qb, 0, band) +
                    geofac_grdiv[:, None, 2, :] * gat(vn, qi, qb, 1, band) +
                    geofac_grdiv[:, None, 3, :] * gat(vn, qi, qb, 2, band) +
                    geofac_grdiv[:, None, 4, :] * gat(vn, qi, qb, 3, band) + tang[:, None, :] * inv_prim[:, None, :] *
                    (gat(zeta, evi, evb, 1, band) - gat(zeta, evi, evb, 0, band)))
            update = np.where(clip_e, difcoef_e * area_edge[:, None, :] * grad, 0.0)
            ddt_vn_apc[:, band, :, t] += np.where(lvl_active[None, :, None], update, 0.0)
