import unittest
from unittest.mock import patch

import numpy as np

from bodies import SoftBody
from physics import PhysicsEngine


class ActivityTests(unittest.TestCase):
    def test_propagation_advances_one_bond_per_pass_in_either_direction(self):
        body = SoftBody('chain', 'armor')
        body.active = np.array([True, False, False, False, False, False, False])
        body.springs = np.array([[0, 1], [2, 1], [2, 3], [4, 3], [4, 5], [6, 5]])

        PhysicsEngine()._propagate_active(body)

        np.testing.assert_array_equal(body.active, [True] * 6 + [False])

    def test_contact_can_wake_and_move_an_inactive_body_in_the_same_substep(self):
        body = SoftBody('point', 'armor')
        body.pos = np.zeros((1, 3))
        body.vel = np.zeros_like(body.pos)
        body.active = np.zeros(1, dtype=bool)
        engine = PhysicsEngine()
        engine.bodies = [body]
        engine.time_scale = 1.0

        wake_on_pass = iter((False, False, False, True))

        def contact(_dt):
            if next(wake_on_pass):
                body.active[0] = True
                body.vel[0, 0] = 2.0

        with patch.object(engine, '_handle_collisions', side_effect=contact):
            engine.step(0.004)

        np.testing.assert_array_equal(body.pos, [[0.002, 0.0, 0.0]])
