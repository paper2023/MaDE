#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

"""Train script for MPEs."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
    #     str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))
    setproctitle.setproctitle("bq-abstract-3")
    
    if not all_args.use_test:
        # run dir
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                    0] + "/results_ours") / all_args.env_name / all_args.experiment_name
        if not run_dir.exists():
            os.makedirs(str(run_dir))

        # wandb
        if all_args.use_wandb:
            run = wandb.init(config=all_args,
                            project=all_args.env_name,
                            entity=all_args.user_name,
                            notes=socket.gethostname(),
                            name=str(all_args.algorithm_name) + "_" +
                            str(all_args.experiment_name) +
                            "_seed" + str(all_args.seed),
                            group=all_args.scenario_name,
                            dir=str(run_dir),
                            job_type="training",
                            reinit=True)
        else:
            curr_run = 'seed_' + str(all_args.seed) + '_'
            if not run_dir.exists():
                curr_run += 'run1'
            else:
                exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
                if len(exst_run_nums) == 0:
                    curr_run += 'run1'
                else:
                    curr_run += 'run%i' % (max(exst_run_nums) + 1)
                    
                #curr_run += '_seed_' + str(all_args.seed)
                
            run_dir = run_dir / curr_run
            if not run_dir.exists():
                os.makedirs(str(run_dir))
            
    else:
        run_dir=''

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_test else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        all_args.use_graph = True #########################here
        if all_args.use_graph:
            from onpolicy.runner.shared.mpe_runner import MPEGraphRunner as Runner
        else:
            from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    if not all_args.use_test:
        runner.run()
    else:
        runner.test()
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if not all_args.use_test:
        if all_args.use_wandb:
            run.finish()
        else:
            runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
            runner.writter.close()


if __name__ == "__main__":
    
    args = ['--env_name', 'MPE', '--algorithm_name', 'rmappo', '--experiment_name', '4agents', '--num_agents', '4', '--num_landmarks', '4', '--scenario_name', 'simple_spread', 
     '--seed', '1', '--n_training_threads', '1', '--n_rollout_threads', '64', '--num_mini_batch', '1', '--episode_length', '25', '--num_env_steps', '5000500', 
     '--ppo_epoch', '10', '--gain', '0.01', '--lr', '7e-4', '--critic_lr', '7e-4', '--use_wandb', 'False']
    # 
    
    agents_lst = ['3', '4', '6', '8', '10']
    eps_lst = ['25', '25', '25', '25', '25']
    # bis_models_local = ['3agents/bis_agent._960000.pt', '4agents/bis_agent._800000.pt', '6agents/bis_agent._2820000.pt', 
    #                     '8agents/bis_agent._3400000.pt', '10agents/bis_agent._5000000.pt']
    # bis_models_global = ['3agents/bis_agent_global._960000.pt', '4agents/bis_agent_global._800000.pt', '6agents/bis_agent_global._2820000.pt', 
    #                      '8agents/bis_agent_global._3400000.pt', '10agents/bis_agent_global._5000000.pt']
    
    ################# 10 agents on-policy-mpe-161-20220301/onpolicy/scripts/results/MPE/simple_spread/rmappo_centroids/10agents/emb64/MI_no_shuffle
    # bis_models_local = ['3agents/bis_agent._960000.pt', '4agents/bis_agent._800000.pt', '6agents/bis_agent._2820000.pt', 
    #                     '8agents/bis_agent._3400000.pt', 'bis_agent._3240000.pt']
    # bis_models_global = ['3agents/bis_agent_global._960000.pt', '4agents/bis_agent_global._800000.pt', '6agents/bis_agent_global._2820000.pt', 
    #                      '8agents/bis_agent_global._3400000.pt', 'bis_agent_global._3240000.pt']
    
    ################# 10 agents on-policy-mpe-161-20220301/onpolicy/scripts/results/MPE/simple_spread/rmappo_centroids/10agents/emb64/no_MI_no_shuffle
    bis_models_local = ['bis_agent._960000.pt', 'bis_agent._800000.pt', 'bis_agent._2820000.pt', 
                        'bis_agent._3400000.pt', 'bis_agent._3640000.pt']
    bis_models_global = ['bis_agent_global._960000.pt', 'bis_agent_global._800000.pt', 'bis_agent_global._2820000.pt', 
                         'bis_agent_global._3400000.pt', 'bis_agent_global._3640000.pt']
    
    ################# 10 agents on-policy-mpe-161-20220301/onpolicy/scripts/results/MPE/simple_spread/rmappo_centroids/10agents/emb64/MI_no_shuffle
    bis_models_local = ['bis_agent._960000.pt', 'bis_agent._800000.pt', 'bis_agent._2820000.pt', 
                        'bis_agent._3400000.pt', 'bis_agent._3240000.pt']
    bis_models_global = ['bis_agent_global._960000.pt', 'bis_agent_global._800000.pt', 'bis_agent_global._2820000.pt', 
                         'bis_agent_global._3400000.pt', 'bis_agent_global._3240000.pt']

    # eps_lst = ['25', '25', '50', '80', '100']
    # model_dirs = ['run1', 'run1', 'run2_eps50', 'run2_eps80', 'run2_eps100']
    # index = 0 # 2 4
    # index = 2 # 2 4
    index = 4 # 2 4
    
    num_agents = agents_lst[index]
    args[21] = eps_lst[index]
    # algorithm_name = 'mappo'
    # algorithm_name = 'rmappo'
    algorithm_name = 'ippo'
    args[3], args[7], args[9] = algorithm_name, num_agents, num_agents #### algorithm_name  num_agents num_landmarks
    
    # experiment_name = num_agents + 'agents_' + args[3] + '_ours_joint' ##### 导入的互模拟模型进行联合训练
    # args.extend(['--use_joint_train', 'True'])
    
    experiment_name = num_agents + 'agents_' + args[3] + '_0318_ippo_thr64' ##### 64 --> 16 
    
    
    args[5] = experiment_name
    
    args.extend(['--use_test', 'False', '--use_graph', 'True', '--use_ReLU', '--global_bis_agents_path', '', '--local_bis_agents_path', ''])
    #################################### L159 all_args.use_graph 重新设置
    
    args[-1] = bis_models_local[index]
    args[-3] = bis_models_global[index]
    
    
    for i in range(0, 5):
    # for i in range(3, 6):
        args[13]=str(i) ### seed
        main(args)
    # main(args)
