# -*- coding: utf-8 -*-
"""Physics engine: time stepping, springs, bond breaking, collisions."""

import numpy as np
from scipy.spatial import cKDTree

from materials import MATERIALS


class PhysicsEngine:
    def __init__(self):
        self.bodies = []
        self.time = 0.0
        self.time_scale = 0.002
        self.collision_dist = 0.012
        self.impact_occurred = False
        self.impact_start_x = None
        self.max_penetration = 0.0
        # Active ERA detonations (one per explosive body, once initiated).
        self.detonations = []
        # Precomputed contact-area constant (depends only on collision_dist).
        self._ca = np.pi * (self.collision_dist * 0.5) ** 2
        # Packed collision world: static arrays live across substeps; pos/vel
        # are copied in each call. Rebuilt only when the body list / materials change.
        self._world = None

    def reset(self):
        self.time = 0.0
        self.impact_occurred = False
        
        self.impact_start_x = None
        self.max_penetration = 0.0
        self._clear_detonations()
        for body in self.bodies:
            body.pos = body.rest_pos.copy()
            body.vel.fill(0)
            body.angular_vel = np.zeros(3, dtype=np.float32)

            if body.static is not None:
                body.pos[body.static] = body.rest_pos[body.static].copy()
                body.vel[body.static] = 0.0

            if body.springs is not None and len(body.springs) > 0:
                i_idx = body.springs[:, 0]
                j_idx = body.springs[:, 1]
                delta = body.pos[j_idx] - body.pos[i_idx]
                body.spring_rest = np.linalg.norm(delta, axis=1).copy()
                body.spring_rest_original = body.spring_rest.copy()

            if body.body_type == 'penetrator':
                body.vel[:, 0] = body.initial_velocity
                if body.active is not None:
                    body.active.fill(True)
            else:
                if body.active is not None:
                    body.active.fill(False)

    def _clear_detonations(self):
        for det in self.detonations:
            body = det['body']
            backup = det.get('static_backup')
            if (backup is not None and body.static is not None
                    and len(backup) == len(body.static)):
                body.static = backup.copy()
            mass_backup = getattr(body, '_mass_backup', None)
            if mass_backup is not None and len(mass_backup) == len(body.mass):
                body.mass = mass_backup.copy()
            body.burnt = None
            body.blast_energy = None
        self.detonations = []

    def _initiate_detonation(self, body_idx, origin):
        """Start a detonation wave. Work on neighbors comes from product pressure
        (chemical energy → P dV), not a prescribed flyer velocity."""
        body = self.bodies[body_idx]
        mat = MATERIALS[body.material]
        rho = float(mat['density'])
        d_cj = float(mat.get('detonation_velocity', 7920.0))
        gamma = float(mat.get('product_gamma', 3.0))
        gurney_v = float(mat.get('gurney_velocity', 2700.0))
        spacing = float(getattr(body, 'particle_spacing', self.collision_dist))
        body.burnt = np.zeros(body.n_particles, dtype=bool)
        body.blast_energy = np.zeros(body.n_particles, dtype=np.float64)
        body._mass_backup = body.mass.copy()
        self.detonations.append({
            'body': body,
            'body_idx': body_idx,
            'origin': np.asarray(origin, dtype=np.float64),
            'radius': 0.0,
            'd_vel': d_cj,
            # Chapman–Jouguet pressure from the product EOS: P = ρ D² / (γ+1)
            'p_cj': rho * d_cj ** 2 / (gamma + 1.0),
            'gamma': gamma,
            'r0': spacing,
            'area': spacing ** 2,
            'e_per_mass': 0.5 * gurney_v ** 2,
            'static_backup': None if body.static is None else body.static.copy(),
        })

    def _ensure_world(self):
        """Build or reuse packed per-particle arrays that do not change mid-step."""
        slices = []
        sig_parts = []
        n_total = 0
        col_dist = self.collision_dist
        for bi, b in enumerate(self.bodies):
            if b.pos is None:
                continue
            n = b.n_particles
            slices.append((bi, n_total, n))
            sig_parts.append((
                id(b), n, id(b.mass), b.material, b.body_type,
                float(getattr(b, 'particle_spacing', col_dist)),
            ))
            n_total += n
        sig = (n_total, col_dist, tuple(sig_parts))
        w = self._world
        if w is not None and w['sig'] == sig:
            return w
        if n_total == 0:
            self._world = None
            return None

        mass = np.empty(n_total, dtype=np.float64)
        body_idx = np.empty(n_total, dtype=np.int32)
        part_idx = np.empty(n_total, dtype=np.int32)
        strength = np.empty(n_total, dtype=np.float64)
        restitution = np.empty(n_total, dtype=np.float64)
        youngs = np.empty(n_total, dtype=np.float64)
        friction = np.empty(n_total, dtype=np.float64)
        init_vel = np.empty(n_total, dtype=np.float64)
        is_pen = np.empty(n_total, dtype=np.int8)
        spacing = np.empty(n_total, dtype=np.float64)
        n_max = 0
        for bi, off, n in slices:
            body = self.bodies[bi]
            sl = slice(off, off + n)
            mat = MATERIALS[body.material]
            mass[sl] = body.mass
            body_idx[sl] = bi
            part_idx[sl] = np.arange(n, dtype=np.int32)
            strength[sl] = mat['strength']
            restitution[sl] = mat['restitution']
            youngs[sl] = mat['youngs_modulus']
            friction[sl] = mat['friction_coeff']
            init_vel[sl] = (mat.get('initiation_velocity', 0.0)
                            if mat.get('explosive') else 0.0)
            is_pen[sl] = 1 if body.body_type == 'penetrator' else 0
            spacing[sl] = getattr(body, 'particle_spacing', col_dist)
            if n > n_max:
                n_max = n

        self._world = {
            'sig': sig,
            'slices': slices,
            'pos': np.empty((n_total, 3), dtype=np.float64),
            'vel': np.empty((n_total, 3), dtype=np.float64),
            'mass': mass,
            'body_idx': body_idx,
            'part_idx': part_idx,
            'strength': strength,
            'restitution': restitution,
            'youngs': youngs,
            'friction': friction,
            'init_vel': init_vel,
            'is_pen': is_pen,
            'spacing': spacing,
            'locked': np.zeros(n_total, dtype=bool),
            'n_max': n_max,
            'spring_enc': {},
            'spring_sig': {},
        }
        return self._world

    def _sync_world_pv(self, w):
        pos, vel = w['pos'], w['vel']
        for bi, off, n in w['slices']:
            body = self.bodies[bi]
            pos[off:off + n] = body.pos
            vel[off:off + n] = body.vel

    def _writeback_world_pv(self, w, write_pos=True, write_vel=True):
        pos, vel = w['pos'], w['vel']
        for bi, off, n in w['slices']:
            body = self.bodies[bi]
            if write_pos:
                body.pos[:] = pos[off:off + n]
            if write_vel:
                body.vel[:] = vel[off:off + n]

    def _cached_spring_encoded(self, w):
        enc = w['spring_enc']
        sigs = w['spring_sig']
        n_max = w['n_max']
        live = {bi for bi, _, _ in w['slices']}
        for bi in list(enc):
            if bi not in live:
                del enc[bi]
                sigs.pop(bi, None)
        for bi, _, _ in w['slices']:
            body = self.bodies[bi]
            springs = body.springs
            if springs is None or len(springs) == 0:
                enc.pop(bi, None)
                sigs.pop(bi, None)
                continue
            ssig = (id(springs), len(springs))
            if sigs.get(bi) == ssig:
                continue
            lo = np.minimum(springs[:, 0], springs[:, 1]).astype(np.int64)
            hi = np.maximum(springs[:, 0], springs[:, 1]).astype(np.int64)
            enc[bi] = lo * n_max + hi
            sigs[bi] = ssig
        return enc

    def _pack_particles(self):
        w = self._ensure_world()
        if w is None:
            return None, None, None, None, None, None
        self._sync_world_pv(w)
        locked = w['locked']
        locked.fill(False)
        for bi, off, n in w['slices']:
            body = self.bodies[bi]
            if body.static is not None:
                locked[off:off + n] = body.static
        return (w['pos'], w['vel'], w['mass'],
                w['body_idx'], w['part_idx'], locked)

    def _unpack_velocities(self, world_vel):
        w = self._world
        if w is None:
            offset = 0
            for b in self.bodies:
                if b.pos is None:
                    continue
                n = b.n_particles
                b.vel = world_vel[offset:offset + n].astype(np.float32)
                offset += n
            return
        if world_vel is not w['vel']:
            w['vel'][...] = world_vel
        self._writeback_world_pv(w, write_pos=False, write_vel=True)

    def _apply_detonations(self, dt):
        """Detonation front + PdV work from the products.

        Newly burnt particles receive their chemical energy (½ m √(2E)²).
        Every substep, remaining energy is spent as pressure on whatever is
        actually nearby: P = P_CJ (r0/r)^γ, F = P A, a = F/m. No flyer list,
        no prescribed plate speed."""
        for det in self.detonations:
            body = det['body']
            burnt = getattr(body, 'burnt', None)
            if body.pos is None or burnt is None or len(burnt) != body.n_particles:
                continue
            if getattr(body, 'blast_energy', None) is None or len(body.blast_energy) != body.n_particles:
                body.blast_energy = np.zeros(body.n_particles, dtype=np.float64)

            det['radius'] += det['d_vel'] * dt
            newly = (np.linalg.norm(body.pos - det['origin'], axis=1) <= det['radius']) & ~burnt
            if np.any(newly):
                burnt |= newly
                orig_m = body._mass_backup[newly].astype(np.float64)
                body.blast_energy[newly] = orig_m * det['e_per_mass']
                if body.springs is not None and len(body.springs) > 0:
                    keep = ~(burnt[body.springs[:, 0]] | burnt[body.springs[:, 1]])
                    body.springs = body.springs[keep]
                    body.spring_rest = body.spring_rest[keep]
                    body.spring_rest_original = body.spring_rest_original[keep]
                    body.spring_stiff = body.spring_stiff[keep]
                if body.static is not None:
                    body.static[newly] = False
                if body.active is not None:
                    body.active[newly] = True

            hot = burnt & (body.blast_energy > 0.0)
            if not np.any(hot):
                continue

            world_pos, world_vel, world_mass, owners, local_i, locked = self._pack_particles()
            if world_pos is None:
                continue

            r0 = det['r0']
            gamma = det['gamma']
            p_cj = det['p_cj']
            area = det['area']
            # Range where P has fallen to ~1% of P_CJ for this EOS — not a flyer radius.
            r_max = r0 * (100.0 ** (1.0 / max(gamma, 1.0)))

            src_idx = np.where(hot)[0]
            src_pos = body.pos[src_idx].astype(np.float64)
            src_e = body.blast_energy[src_idx]
            tree = cKDTree(world_pos)
            hits = tree.query_ball_point(src_pos, r_max)
            exp_bi = det['body_idx']
            world_of = np.where(owners == exp_bi)[0]
            local_of = local_i[world_of]
            local_to_world = np.empty(body.n_particles, dtype=np.int32)
            local_to_world[local_of] = world_of

            for k, nbrs in enumerate(hits):
                if src_e[k] <= 0.0 or not nbrs:
                    continue
                nbrs = np.asarray(nbrs, dtype=np.int32)
                delta = world_pos[nbrs] - src_pos[k]
                dist = np.linalg.norm(delta, axis=1)
                live = (dist > 1e-6) & ~locked[nbrs]
                same_src = (owners[nbrs] == exp_bi) & (local_i[nbrs] == src_idx[k])
                live &= ~same_src
                same_body = owners[nbrs] == exp_bi
                if np.any(same_body):
                    live[same_body] &= burnt[local_i[nbrs[same_body]]]
                if not np.any(live):
                    continue
                nbrs, delta, dist = nbrs[live], delta[live], dist[live]
                nrm = delta / dist[:, np.newaxis]
                P = p_cj * (r0 / np.maximum(dist, r0)) ** gamma
                F = P * area
                m_j = np.maximum(world_mass[nbrs], 1e-12)
                m_i = max(float(world_mass[local_to_world[src_idx[k]]]), 1e-12)
                rel = world_vel[nbrs] - world_vel[local_to_world[src_idx[k]]]
                v_out = np.einsum('ij,ij->i', rel, nrm)
                work = np.maximum(F * v_out, 0.0) * dt
                work_sum = float(np.sum(work))
                if work_sum <= 1e-18:
                    p_imp = F * dt
                    m_red = 1.0 / (1.0 / m_i + 1.0 / m_j)
                    work_sum = float(np.sum(0.5 * p_imp ** 2 / np.maximum(m_red, 1e-12)))
                if work_sum > src_e[k] and work_sum > 1e-18:
                    F = F * (src_e[k] / work_sum)
                    work_sum = src_e[k]
                dv = (F * dt / m_j)[:, np.newaxis] * nrm
                world_vel[nbrs] += dv
                world_vel[local_to_world[src_idx[k]]] -= (dv * m_j[:, np.newaxis]).sum(axis=0) / m_i
                src_e[k] = max(src_e[k] - work_sum, 0.0)
                for bi in np.unique(owners[nbrs]):
                    other = self.bodies[int(bi)]
                    if other.active is not None:
                        sel = nbrs[owners[nbrs] == bi]
                        other.active[local_i[sel]] = True

            body.blast_energy[src_idx] = src_e
            self._unpack_velocities(world_vel)

    def _propagate_active(self, body):
        if body.springs is None or len(body.springs) == 0:
            return
        i_idx = body.springs[:, 0]
        j_idx = body.springs[:, 1]
        for _ in range(5):
            new_i = body.active[j_idx] & ~body.active[i_idx]
            new_j = body.active[i_idx] & ~body.active[j_idx]
            if not (np.any(new_i) or np.any(new_j)):
                break
            if np.any(new_i):
                body.active[i_idx[new_i]] = True
            if np.any(new_j):
                body.active[j_idx[new_j]] = True

    def step(self, dt):
        real_dt = dt * self.time_scale
        substeps = 4
        sub_dt = real_dt / substeps

        for _ in range(substeps):
            for body in self.bodies:
                if body.active is None:
                    continue
                if body.body_type == 'penetrator':
                    body.active.fill(True)
                else:
                    vel_sq = np.sum(body.vel ** 2, axis=1)
                    body.active = vel_sq > 0.01
                if body.static is not None:
                    body.active[body.static] = False
                    body.vel[body.static] = 0.0

            self._handle_collisions(sub_dt)

            if self.detonations:
                self._apply_detonations(sub_dt)

            for body in self.bodies:
                if body.active is not None:
                    self._propagate_active(body)

            for body in self.bodies:
                self._apply_springs(body, sub_dt)

            for body in self.bodies:
                mask = body.active if body.active is not None else np.ones(body.n_particles, dtype=bool)
                if body.static is not None:
                    mask = mask & ~body.static
                body.pos[mask] += body.vel[mask] * sub_dt

                # Rotational integration: rotate particles around CoM
                omega = body.angular_vel
                omega_sq = float(np.dot(omega, omega))
                if omega_sq > 1e-12 and body.springs is not None and len(body.springs) > 0:
                    connected = np.zeros(body.n_particles, dtype=bool)
                    connected[body.springs[:, 0]] = True
                    connected[body.springs[:, 1]] = True
                    m_conn = body.mass[connected]
                    total_m = float(np.sum(m_conn))
                    if total_m > 1e-12:
                        com = np.sum(body.pos[connected] * m_conn[:, np.newaxis], axis=0) / total_m
                        r = body.pos - com
                        rot_disp = np.cross(omega, r) * sub_dt
                        body.pos[mask] += rot_disp[mask]

        self.time += real_dt
        stats = {'time_us': self.time * 1e6, 'vel': 0, 'pen': 0,
                 'detonated': bool(self.detonations)}
        for b in self.bodies:
            if b.body_type == 'penetrator':
                stats['vel'] = np.mean(np.sqrt(np.sum(b.vel ** 2, axis=1)))
                if self.impact_occurred and self.impact_start_x is not None:
                    current_pen = (np.max(b.pos[:, 0]) - self.impact_start_x) * 1000
                    self.max_penetration = max(self.max_penetration, current_pen)
                    stats['pen'] = self.max_penetration
        return stats

    def _apply_springs(self, body, dt):
        if body.springs is None or len(body.springs) == 0:
            return

        if body.active is not None:
            si = body.springs[:, 0]
            sj = body.springs[:, 1]
            active_mask = body.active[si] | body.active[sj]
            if not np.any(active_mask):
                return
        else:
            active_mask = np.ones(len(body.springs), dtype=bool)

        a_springs = body.springs[active_mask]
        a_rest = body.spring_rest[active_mask]
        a_rest_orig = body.spring_rest_original[active_mask]
        a_stiff = body.spring_stiff[active_mask]
        i_idx = a_springs[:, 0]
        j_idx = a_springs[:, 1]

        delta = body.pos[j_idx] - body.pos[i_idx]
        lengths = np.linalg.norm(delta, axis=1)
        lengths = np.maximum(lengths, 1e-6)
        dirs = delta / lengths[:, np.newaxis]
        strain = (lengths - a_rest) / a_rest
        abs_strain = np.abs(strain)

        mat = MATERIALS[body.material]
        breaking_strain = mat['breaking_strain']
        compression_limit = mat['compression_limit']

        # Break based on total elongation from the ORIGINAL rest length, not the
        # plastic rest length. Plasticity ratchets spring_rest, so strain vs spring_rest
        # stays near yield_s and never reaches breaking_strain without this.
        strain_from_orig = (lengths - a_rest_orig) / a_rest_orig
        broken = strain_from_orig > breaking_strain
        comp_break = mat.get('compression_breaking_strain', None)
        if comp_break is not None:
            broken |= (strain_from_orig < -comp_break)
        if np.any(broken):
            broken_full = np.where(active_mask)[0][np.where(broken)[0]]
            keep = np.ones(len(body.springs), dtype=bool)
            keep[broken_full] = False
            body.springs = body.springs[keep]
            body.spring_rest = body.spring_rest[keep]
            body.spring_rest_original = body.spring_rest_original[keep]
            body.spring_stiff = body.spring_stiff[keep]

            if len(body.springs) == 0:
                body.angular_vel = np.zeros(3, dtype=np.float32)
                return

            if body.active is not None:
                si2 = body.springs[:, 0]
                sj2 = body.springs[:, 1]
                active_mask = body.active[si2] | body.active[sj2]
            else:
                active_mask = np.ones(len(body.springs), dtype=bool)
            if not np.any(active_mask):
                return

            a_springs = body.springs[active_mask]
            a_rest = body.spring_rest[active_mask]
            a_rest_orig = body.spring_rest_original[active_mask]
            a_stiff = body.spring_stiff[active_mask]
            i_idx = a_springs[:, 0]
            j_idx = a_springs[:, 1]
            delta = body.pos[j_idx] - body.pos[i_idx]
            lengths = np.linalg.norm(delta, axis=1)
            lengths = np.maximum(lengths, 1e-6)
            dirs = delta / lengths[:, np.newaxis]
            strain = (lengths - a_rest) / a_rest
            abs_strain = np.abs(strain)

        force_mag = a_stiff * strain
        forces = force_mag[:, np.newaxis] * dirs

        # Progressive compressive hardening.
        # Penetrators: all springs use material spring_min_ratio (keeps core compact).
        # Armor: only diagonal springs get the floor (prevents lattice inversion).
        #   Axial springs on armor must be free to compress so the penetrator can
        #   plastically deform and fracture them — adding a floor here causes elastic
        #   bounce-back instead of plastic failure, killing penetration depth.
        mat_floor = mat.get('spring_min_ratio', 0.75)
        spacing = getattr(body, 'particle_spacing', 0.012)
        diag_mask = a_rest > spacing * 1.1
        if body.body_type == 'penetrator':
            per_min = np.full(len(a_rest), mat_floor)
        else:
            per_min = np.where(diag_mask, mat_floor, 0.0)

        harden_candidates = (strain < 0) & (per_min > 0)
        if np.any(harden_candidates):
            mc = harden_candidates
            max_cs = -(1.0 - per_min[mc])
            frac = np.clip(strain[mc] / max_cs, 0.0, 0.98)
            hardening = 1.0 + frac ** 2 / (1.0 - frac + 0.3)  # floor 0.3 caps ~4×; 1e-3 allowed 47× which exceeded critical sub_dt for stiff ceramics
            force_mag[mc] *= hardening
            forces[mc] = force_mag[mc, np.newaxis] * dirs[mc]

        # Soft repulsion backstop for armor bodies. Penetrators skip this: their
        # progressive hardening (spring_min_ratio) engages at ~5-12% compression,
        # far before compression_limit (40-55%) would ever be reached.
        if body.body_type != 'penetrator':
            ratio = lengths / a_rest
            rep_mask = ratio < compression_limit
            if np.any(rep_mask):
                extra = (compression_limit - ratio[rep_mask]) * a_stiff[rep_mask] * 3.0
                forces[rep_mask] -= extra[:, np.newaxis] * dirs[rep_mask]

        # Plastic deformation: rest length ratchets to follow deformation in both
        # directions. Springs resist further change from their current plastic state,
        # not from their original length. A small elastic tail (yield_s) is preserved
        # so there is still a restoring force at the new deformed state.
        yield_s = mat['yield_strain']

        comp_plastic = strain < -yield_s
        if np.any(comp_plastic):
            p_idx = np.where(active_mask)[0][comp_plastic]
            new_rest = lengths[comp_plastic] / (1.0 - yield_s)
            further = new_rest < body.spring_rest[p_idx]
            if np.any(further):
                body.spring_rest[p_idx[further]] = new_rest[further]

        tens_plastic = strain > yield_s
        if np.any(tens_plastic):
            p_idx = np.where(active_mask)[0][tens_plastic]
            new_rest = lengths[tens_plastic] / (1.0 + yield_s)
            # Only ratchet if stretched further than ever before.
            further = new_rest > body.spring_rest[p_idx]
            if np.any(further):
                body.spring_rest[p_idx[further]] = new_rest[further]

        # np.bincount is significantly faster than np.add.at for force scatter.
        n = body.n_particles
        force_accum = np.empty((n, 3), dtype=np.float32)
        for d in range(3):
            force_accum[:, d] = (
                np.bincount(i_idx, weights=forces[:, d], minlength=n)
                - np.bincount(j_idx, weights=forces[:, d], minlength=n)
            )

        if body.active is not None:
            if body.static is not None:
                update = body.active & ~body.static
            else:
                update = body.active
        else:
            update = ~body.static if body.static is not None else np.ones(body.n_particles, dtype=bool)

        nz = update & (body.mass > 0)
        body.vel[nz] += force_accum[nz] / body.mass[nz, np.newaxis] * dt

        connected = np.zeros(body.n_particles, dtype=bool)
        if len(body.springs) > 0:
            connected[body.springs[:, 0]] = True
            connected[body.springs[:, 1]] = True
        m_conn = body.mass[connected]
        total_m = np.sum(m_conn)
        if total_m > 1e-12:
            com = np.sum(body.pos[connected] * m_conn[:, np.newaxis], axis=0) / total_m
            r = body.pos - com
            torque = np.sum(np.cross(r, force_accum), axis=0)
            I = max(np.sum(m_conn * np.sum((body.pos[connected] - com) ** 2, axis=1)), 1e-12)
            body.angular_vel += (torque / I).astype(np.float32) * dt
            body.angular_vel *= 0.7

    def _handle_collisions(self, dt):
        bodies = self.bodies
        col_dist = self.collision_dist
        w = self._ensure_world()
        if w is None:
            return

        self._sync_world_pv(w)
        all_pos = w['pos']
        all_vel = w['vel']
        all_mass = w['mass']
        all_body_idx = w['body_idx']
        all_part_idx = w['part_idx']
        p_strength = w['strength']
        p_restitution = w['restitution']
        p_youngs = w['youngs']
        p_friction = w['friction']
        p_init_vel = w['init_vel']
        p_is_pen = w['is_pen']
        p_spacing = w['spacing']

        # --- Pair finding: cKDTree is C-level, far faster than Python cell loop ---
        tree = cKDTree(all_pos)
        pairs = tree.query_pairs(col_dist, output_type='ndarray')

        if len(pairs) == 0:
            return

        ii = pairs[:, 0]
        jj = pairs[:, 1]
        bi_arr = all_body_idx[ii]
        bj_arr = all_body_idx[jj]
        same_body = bi_arr == bj_arr

        # Per-pair effective collision distance (intra-body uses particle spacing)
        eff_cd = np.where(same_body, p_spacing[ii] * 0.85, col_dist)

        diff = all_pos[jj] - all_pos[ii]          # (M, 3)
        dist_sq = np.einsum('ij,ij->i', diff, diff)

        valid = (dist_sq < eff_cd ** 2) & (dist_sq > 1e-12)

        # Spring-connected same-body pairs: the spring already handles their interaction.
        # Applying a collision impulse on top would destabilise the spring and blow off
        # closely-spaced particles like nose-tip steps. Skip those pairs here; once a
        # spring breaks it is removed from body.springs, so the pair re-enters collision.
        sb_idx = np.where(same_body & valid)[0]
        if len(sb_idx):
            n_max = w['n_max']
            body_spring_encoded = self._cached_spring_encoded(w)
            li_arr = all_part_idx[ii[sb_idx]]
            lj_arr = all_part_idx[jj[sb_idx]]
            bk_arr = bi_arr[sb_idx]
            lo_pairs = np.minimum(li_arr, lj_arr).astype(np.int64)
            hi_pairs = np.maximum(li_arr, lj_arr).astype(np.int64)
            pair_encoded = lo_pairs * n_max + hi_pairs
            for bi, encoded in body_spring_encoded.items():
                body_mask = bk_arr == bi
                if not np.any(body_mask):
                    continue
                is_spring = np.isin(pair_encoded[body_mask], encoded)
                valid[sb_idx[body_mask][is_spring]] = False
        if not np.any(valid):
            return

        ii = ii[valid]; jj = jj[valid]
        bi_arr = bi_arr[valid]; bj_arr = bj_arr[valid]
        same_body = same_body[valid]
        eff_cd = eff_cd[valid]
        diff = diff[valid]
        dist_sq = dist_sq[valid]

        dist = np.sqrt(dist_sq)
        normal = diff / dist[:, np.newaxis]
        overlap = eff_cd - dist

        rel_vel = all_vel[ii] - all_vel[jj]
        vel_normal = np.einsum('ij,ij->i', rel_vel, normal)

        m1 = all_mass[ii]; m2 = all_mass[jj]
        reduced_mass = m1 * m2 / (m1 + m2)
        s1 = p_strength[ii]; s2 = p_strength[jj]
        e1 = p_restitution[ii]; e2 = p_restitution[jj]
        mu = (p_friction[ii] + p_friction[jj]) * 0.5
        E_avg = (p_youngs[ii] + p_youngs[jj]) * 0.5

        s_sum  = s1 + s2
        ratio1 = s2 / s_sum
        ratio2 = s1 / s_sum

        # Impact detection (pen + armor pair with different body indices)
        is_pen_armor = ((p_is_pen[ii] + p_is_pen[jj]) == 1) & (~same_body)
        if np.any(is_pen_armor) and not self.impact_occurred:
            self.impact_occurred = True
            for body in self.bodies:
                if body.body_type == 'armor':
                    self.impact_start_x = np.min(body.pos[:, 0])
                    break

        # --- ERA initiation: charge shocked hard enough by whatever reaches it ---
        # Gated on the impactor's own speed, not the pair separation speed: shock
        # initiation is driven by impact velocity, and a charge merely resting against
        # its cassette plates must never light off.
        thresh = np.maximum(p_init_vel[ii], p_init_vel[jj])
        cand = np.where((thresh > 0) & (~same_body) & (vel_normal > 0))[0]
        if len(cand):
            exp_is_i = p_init_vel[ii[cand]] > 0
            striker = np.where(exp_is_i, jj[cand], ii[cand])
            hot = cand[np.linalg.norm(all_vel[striker], axis=1) > thresh[cand]]
            if len(hot):
                already = {d['body_idx'] for d in self.detonations}
                exp_body = np.where(p_init_vel[ii[hot]] > 0, bi_arr[hot], bj_arr[hot])
                for exp_bi in np.unique(exp_body):
                    if exp_bi in already:
                        continue
                    first = hot[exp_body == exp_bi][0]
                    self._initiate_detonation(
                        int(exp_bi), 0.5 * (all_pos[ii[first]] + all_pos[jj[first]]))

        # --- Position corrections (vectorized) ---
        contact_stiff = np.clip(np.sqrt(s_sum / (2.0 * 800e6)), 0.5, 1.5)
        half_stiff = 0.5 * contact_stiff
        scale_i = (overlap * ratio1 * half_stiff)[:, np.newaxis]
        scale_j = (overlap * ratio2 * half_stiff)[:, np.newaxis]
        np.add.at(all_pos, ii, -normal * scale_i)
        np.add.at(all_pos, jj,  normal * scale_j)

        # --- Velocity updates: approaching pairs ---
        app = vel_normal > 0
        if np.any(app):
            a = np.where(app)[0]
            base_e = np.sqrt(e1[a] * e2[a])
            v_ratio = np.minimum(vel_normal[a] / 1000.0, 2.0)
            # Lower restitution floor for pen-armor: these are penetrating contacts,
            # not elastic bounces. The 0.2 floor was adding fake bounce and
            # over-decelerating the penetrator. Metal-on-metal penetration: ~0.02.
            rest_min = np.where(is_pen_armor[a], 0.02, 0.2)
            rest = np.maximum(rest_min, base_e * (1.0 - 0.3 * v_ratio ** 2))
            sr = np.minimum(s1[a], s2[a]) / np.maximum(s1[a], s2[a])
            loss = np.minimum(0.5, (1.0 - sr) * 0.4
                              + np.minimum(0.15, mu[a] * vel_normal[a] / 2000.0))
            eff_e = np.maximum(0.2, np.minimum(rest * (1.0 - loss), base_e))
            imp = np.minimum((1.0 + eff_e) * reduced_mass[a] * vel_normal[a],
                             reduced_mass[a] * vel_normal[a] * 1.5)
            imp = np.maximum(0.0, imp)
            # Pen-armor impulse scale: inversely proportional to vel_normal so that
            # oblique impacts (low vel_normal due to cos θ) keep the full 0.4× while
            # perpendicular impacts (high vel_normal) are scaled down proportionally.
            # Pivot at 400 m/s: contacts below this get 0.4×, above this scale as
            # 0.4 × 400 / vel_normal. Prevents perpendicular over-resistance without
            # touching the angled (60°) calibration where vel_normal ≈ 395–400 m/s.
            pen_armor_scale = np.where(
                is_pen_armor[a],
                np.minimum(0.4, 0.4 * 400.0 / np.maximum(vel_normal[a], 400.0)),
                1.0)
            intra_scale = np.where(same_body[a],
                                   np.where(p_is_pen[ii[a]] == 1, 0.85, 0.9),
                                   pen_armor_scale)
            imp *= intra_scale
            impulse = imp[:, np.newaxis] * normal[a]
            np.add.at(all_vel, ii[a], -impulse / m1[a, np.newaxis])
            np.add.at(all_vel, jj[a],  impulse / m2[a, np.newaxis])

        # --- Velocity updates: receding / compressed pairs ---
        rec = ~app
        if np.any(rec):
            r = np.where(rec)[0]
            s_ov = overlap[r] / np.maximum(eff_cd[r], 1e-6)
            same_r = same_body[r]
            rf_same = np.minimum(E_avg[r] * s_ov * self._ca * 1e-6, reduced_mass[r] * 200.0)
            rf_diff = np.minimum(overlap[r] * 50.0, reduced_mass[r] * 50.0)
            rf = np.where(same_r, rf_same, rf_diff)
            imp_s = rf[:, np.newaxis] * normal[r] * dt
            mag = np.linalg.norm(imp_s, axis=1, keepdims=True)
            max_vc = (np.where(same_r, 10.0, 5.0) * reduced_mass[r])[:, np.newaxis]
            scale = np.where(mag > 1e-6, np.minimum(1.0, max_vc / np.maximum(mag, 1e-12)), 0.0)
            imp_s *= scale
            np.add.at(all_vel, ii[r], -imp_s / m1[r, np.newaxis])
            np.add.at(all_vel, jj[r],  imp_s / m2[r, np.newaxis])

        # Mark colliding particles active (vectorized per-body grouping)
        for body_id in np.unique(bi_arr):
            body = bodies[body_id]
            if body.active is not None:
                body.active[all_part_idx[ii[bi_arr == body_id]]] = True
                body.active[all_part_idx[jj[bj_arr == body_id]]] = True

        self._writeback_world_pv(w)
