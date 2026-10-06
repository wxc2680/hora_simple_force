from pathlib import Path
import sys
import unittest

from omegaconf import OmegaConf
import torch


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from hora.algo.models.models import ActorCritic


class HoraTactileObservationTest(unittest.TestCase):
    def setUp(self):
        self.task = OmegaConf.load(
            _ROOT / 'configs' / 'task' / 'SimpleHandHora.yaml'
        )
        self.batch_size = 2
        self.obs_dim = self.task.env.numObservations
        self.tactile_dim = (
            len(self.task.env.tactile.linkNames)
            * self.task.env.tactile.rows
            * self.task.env.tactile.columns
        )

    def _make_model(self, proprio_adapt):
        return ActorCritic(
            {
                'actor_units': [512, 256, 128],
                'priv_mlp_units': [256, 128, 8],
                'actions_num': self.task.env.numActions,
                'input_shape': (self.obs_dim,),
                'priv_info': True,
                'proprio_adapt': proprio_adapt,
                'priv_info_dim': self.task.env.hora.privInfoDim,
            }
        )

    def test_task_observation_is_proprioception_plus_tactile(self):
        self.assertEqual(self.tactile_dim, 144)
        self.assertEqual(self.obs_dim, 96 + self.tactile_dim)
        self.assertEqual(self.task.env.hora.proprioDim, 32)

    def test_stage_one_actor_accepts_full_tactile_observation(self):
        model = self._make_model(proprio_adapt=False)
        self.assertEqual(model.actor_mlp.mlp[0].in_features, self.obs_dim + 8)
        action = model.act_inference(
            {
                'obs': torch.randn(self.batch_size, self.obs_dim),
                'priv_info': torch.randn(
                    self.batch_size, self.task.env.hora.privInfoDim
                ),
            }
        )
        self.assertEqual(
            tuple(action.shape), (self.batch_size, self.task.env.numActions)
        )

    def test_stage_two_actor_accepts_full_tactile_observation(self):
        model = self._make_model(proprio_adapt=True)
        self.assertEqual(model.actor_mlp.mlp[0].in_features, self.obs_dim + 8)
        action = model.act_inference(
            {
                'obs': torch.randn(self.batch_size, self.obs_dim),
                'proprio_hist': torch.randn(
                    self.batch_size, self.task.env.hora.propHistoryLen, 32
                ),
            }
        )
        self.assertEqual(
            tuple(action.shape), (self.batch_size, self.task.env.numActions)
        )


if __name__ == '__main__':
    unittest.main()
