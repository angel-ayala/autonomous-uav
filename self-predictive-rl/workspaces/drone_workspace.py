import random
import torch
import time
import numpy as np
from natsort import natsorted
from pathlib import Path

from utils.env import save_frames_as_gif
from utils import logger
from workspaces.common import make_agent, make_env
from utils.misc import DroneEnvMonitor
from utils.misc import args2target
from utils.misc import evaluate_agent


class DroneWorkspace:
    def __init__(self, cfg):
        self.work_dir = Path(cfg.save_dir)
        self.cfg = cfg
        if self.cfg.save_snapshot:
            self.checkpoint_path = self.work_dir / "agents"
            self.checkpoint_path.mkdir(exist_ok=True)
        self.device = torch.device(cfg.device)
        self.set_seed()
        self.train_env = make_env(self.cfg)
        self.train_env = DroneEnvMonitor(self.train_env,
                                         store_path=self.work_dir /'history_training.csv',
                                         n_sensors=4)
        self.agent = make_agent(self.train_env, self.device, self.cfg)
        self._train_step = 0
        self._train_episode = 0
        self._best_eval_returns = -np.inf

    def set_seed(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

    def train(self):
        self._explore()
        self.train_env.init_store()
        self.train_env.set_learning()
        self.train_env.set_episode()

        state, info = self.train_env.reset()
        done, episode_start_time = False, time.time()
        ep_reward, ep_length = 0, 0

        # for _ in range(1, self.cfg.num_train_steps - self.cfg.explore_steps + 1):
        for _ in range(1, self.cfg.num_train_steps + 1):
            action = self.agent.get_action(state, self._train_step)[0]
            next_state, reward, done, truncated, info = self.train_env.step(
                action * self.train_env.action_space.high)
            ep_reward += reward
            ep_length += 1
            self._train_step += 1

            self.agent.env_buffer.push(
                (
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )
            )

            self.agent.update(self._train_step)

            if (self._train_step) % self.cfg.eval_episode_interval == 0:
                self._eval()
                truncated = True
            if (
                self.cfg.save_snapshot
                and (self._train_step) % self.cfg.save_snapshot_interval == 0
            ):
                self.save_snapshot()
                self.train_env.set_episode()
                truncated = True

            if done or truncated:
                self._train_episode += 1
                print(
                    "TRAIN Episode: {}, total numsteps: {}({}), return: {}".format(
                        self._train_episode,
                        self._train_step,
                        ep_length,
                        round(ep_reward, 2),
                    )
                )
                episode_metrics = dict()
                episode_metrics["train/length"] = ep_length
                episode_metrics["train/return"] = ep_reward
                episode_metrics["FPS"] = ep_length / (
                    time.time() - episode_start_time
                )
                # episode_metrics["env_buffer_length"] = len(self.agent.env_buffer)
                logger.record_step("env_steps", self._train_step)
                for k, v in episode_metrics.items():
                    logger.record_tabular(k, v)
                logger.dump_tabular()

                state, info = self.train_env.reset()
                done, episode_start_time = False, time.time()
                ep_reward, ep_length = 0, 0
            else:
                state = next_state

        self.train_env.close()

    def _explore(self):
        state, info = self.train_env.reset()

        for _ in range(1, self.cfg.explore_steps):
            action = self.train_env.action_space.sample()
            next_state, reward, done, truncated, info = self.train_env.step(action)
            self.agent.env_buffer.push(
                (
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )
            )

            if done or truncated:
                state, info = self.train_env.reset()
            else:
                state = next_state

    def _agent_select_action(self, observations):
        action = self.agent.get_action(observations, self._train_step, eval=True)[0]
        return action * self.train_env.action_space.high

    def evaluate(self, episode=None):
        if episode is None:
            episode = self.cfg.test_episode
        logs_path = Path(self.cfg.save_dir)
        agent_models = natsorted(logs_path.glob('agents/*.pt'), key=str)

        for log_ep, agent_path in enumerate(agent_models):
            if episode > -1 and log_ep != episode:
                continue
            elif episode == -1 and log_ep not in [4, 9, 19, 34, 49]:
                continue

            print('Evaluating', agent_path)
            self.agent.load_save_dict(torch.load(agent_path))
            # Target position for evaluation
            targets_pos = args2target(self.train_env, self.cfg.target_pos)
            # Log eval data
            csv_path = logs_path / 'eval' / f"history_{agent_path.stem}.csv"
            csv_path.parent.mkdir(exist_ok=True)
            self.train_env._data_store.store_path = csv_path
            self.train_env.init_store()
            self.train_env.set_episode(log_ep)

            # Iterate over goal position
            for tpos in targets_pos:
                self._eval(tpos)
        self.train_env.close()

    def _eval(self, target_pos=None):
        self.train_env.set_eval()

        ep_reward, ep_steps, elapsed_time = evaluate_agent(
            self._agent_select_action,
            self.train_env,
            self.cfg.num_eval_episodes,
            self.cfg.num_eval_steps,
            target_pos)

        eval_metrics = dict()
        eval_metrics["return"] = ep_reward
        eval_metrics["length"] = ep_steps

        if (
            self.cfg.save_snapshot
            and ep_reward >= self._best_eval_returns
        ):
            self.save_snapshot(best=True)
            self._best_eval_returns = ep_reward

        logger.record_step("eval_steps", ep_steps)
        for k, v in eval_metrics.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()

    def _render_episodes(self, record):
        frames = []
        done = False
        state = self.eval_env.reset()
        while not done:
            action = self.agent.get_action(state, self._train_step, True)
            next_state, _, done, info = self.eval_env.step(action)
            self.eval_env.render()
            state = next_state
        if record:
            save_frames_as_gif(frames)
        print(
            "Episode: {}, episode steps: {}, episode returns: {}".format(
                i, info["episode"]["l"], round(info["episode"]["r"], 2)
            )
        )

    def _eval_bias(self):
        final_mc_list, final_obs_list, final_act_list = self._mc_returns()
        final_mc_norm_list = np.abs(final_mc_list.copy())
        final_mc_norm_list[final_mc_norm_list < 10] = 10

        obs_tensor = torch.FloatTensor(final_obs_list).to(self.device)
        acts_tensor = torch.FloatTensor(final_act_list).to(self.device)
        lower_bound = self.agent.get_lower_bound(obs_tensor, acts_tensor)

        bias = final_mc_list - lower_bound
        normalized_bias_per_state = bias / final_mc_norm_list

        # metrics = dict()
        # metrics["mean_bias"] = np.mean(bias)
        # metrics["std_bias"] = np.std(bias)
        # metrics["mean_normalised_bias"] = np.mean(normalized_bias_per_state)
        # metrics["std_normalised_bias"] = np.std(normalized_bias_per_state)

    def _mc_returns(self):
        final_mc_list = np.zeros(0)
        final_obs_list = []
        final_act_list = []
        n_mc_eval = 1000
        n_mc_cutoff = 350

        while final_mc_list.shape[0] < n_mc_eval:
            o, i = self.eval_env.reset()
            reward_list, obs_list, act_list = [], [], []
            r, d, ep_ret, ep_len = 0, False, 0, 0
            t = False

            while not d or not t:
                a = self.agent.get_action(o, self._train_step, True)
                a *= self.train_env.action_space.high
                obs_list.append(o)
                act_list.append(a)
                o, r, d, t, _ = self.train_env.step(a)
                ep_ret += r
                ep_len += 1
                reward_list.append(r)

            discounted_return_list = np.zeros(ep_len)
            for i_step in range(ep_len - 1, -1, -1):
                if i_step == ep_len - 1:
                    discounted_return_list[i_step] = reward_list[i_step]
                else:
                    discounted_return_list[i_step] = (
                        reward_list[i_step]
                        + self.cfg.gamma * discounted_return_list[i_step + 1]
                    )

            final_mc_list = np.concatenate(
                (final_mc_list, discounted_return_list[:n_mc_cutoff])
            )
            final_obs_list += obs_list[:n_mc_cutoff]
            final_act_list += act_list[:n_mc_cutoff]

        return final_mc_list, np.array(final_obs_list), np.array(final_act_list)

    def save_snapshot(self, best=False):
        if best:
            snapshot = Path(self.checkpoint_path) / "best.pt"
        else:
            snapshot = Path(self.checkpoint_path) / (str(self._train_step) + ".pt")
        save_dict = self.agent.get_save_dict()
        torch.save(save_dict, snapshot)
