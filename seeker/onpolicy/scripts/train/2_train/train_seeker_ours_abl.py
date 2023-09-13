#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
setproctitle.setproctitle("test")

import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.seeker.seek_modify_env import seek_env

from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

"""Train script for MPEs."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
                env.seed(all_args.seed + rank * 1000)
                
            elif all_args.env_name == "seeker":
                env = seek_env(all_args.num_agents, all_args.episode_length)
                
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)], all_args=all_args)


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            elif all_args.env_name == "seeker":
                env = seek_env(all_args.num_agents, all_args.episode_length)
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
        ######################## ippo_without_rnn
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
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
    import setproctitle
    setproctitle.setproctitle("test")
    
    if not all_args.use_test:
        # run dir
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                    0] + "/results_ours_abl") / all_args.env_name / all_args.algorithm_name/ all_args.experiment_name
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


if __name__ == "__main__": # /data/jqruan/clustering-162/clustering/on-policy-seeker-164-20230830-abl/onpolicy/scripts/train/
    # for i in range(0, 5):   ### threads 64
    # for i in range(0, 3):   ### threads 64
    for i in range(3, 6):   ### threads 64
        args = ['--env_name', 'seeker', '--algorithm_name', 'rmappo', '--experiment_name', '4agents', '--num_agents', '4', '--num_landmarks', '4', '--scenario_name', 'simple_spread', 
        '--seed', '1', '--n_training_threads', '8', '--n_rollout_threads', '64', '--num_mini_batch', '4', '--episode_length', '50', '--num_env_steps', '6000000', 
        '--ppo_epoch', '10', '--gain', '0.01', '--lr', '7e-4', '--critic_lr', '7e-4', '--use_wandb', 'False']
        # 
        import sys
        # print(sys.argv)
        index, abl_index = int(sys.argv[1]), int(sys.argv[2])
        # index, abl_index = 2, 0
        # index, abl_index = 2, 1
        # index, abl_index = 2, 2
        # index, abl_index = 3, 0
        # index, abl_index = 3, 1
        # index, abl_index = 3, 2
        # import pdb
        # pdb.set_trace()
        
        
        agents_lst = ['3', '5', '8', '12']
        num_agents = agents_lst[index]
        
        algorithm_name = 'mappo'
        # algorithm_name = 'rmappo'
        # algorithm_name = 'ippo'
        
        ################# 10 agents on-policy-mpe-161-20220301/onpolicy/scripts/results/MPE/simple_spread/rmappo_centroids/10agents/emb64/MI_no_shuffle
        root_path = '/data/jqruan/clustering-162/clustering/on-policy-seeker-164-20230830-abl/onpolicy/scripts/train/results_ours_logs/seeker/'

        agents_path = [
            '/3agents/3agents_mappo/seed_1_run1/models/train_state_logs/emb_64_files_epo_no_global/ensemble_use_MI__shuffle_[04-12]09.46.21/models_ensemble/',
            '/5agents/5agents_mappo/seed_1_run1/models/train_state_logs/emb_64/ensemble__[05-10]11.36.33/models_ensemble/',
            '/8agents/8agents_rmappo/seed_1_run1/models/train_state_logs/emb_64/ensemble__[04-27]14.16.19/models_ensemble/',
            '/12agents/12agents_rmappo/seed_1_run1/models/train_state_logs/emb_64/ensemble_use_MI__shuffle_[04-23]12.48.10/models_ensemble/'
        ]
        if num_agents == '5' or num_agents == '8':
            args.extend(['--cluster_root_path', root_path+agents_path[index]+'no_MI_no_shuffle/']) ## only 5 \ 8agents no_MI_no_shuffle
        else:
            args.extend(['--cluster_root_path', root_path+agents_path[index]+'MI_shuffle/'])
        
        bis_models_local = [
            root_path+agents_path[index]+'bis_agent._2500000.pt', 
            root_path+agents_path[index]+'bis_agent._60000.pt',
            root_path+agents_path[index]+'bis_agent._800000.pt',
            root_path+agents_path[index]+'bis_agent._2180000.pt'
            ]  
        bis_models_global = [
            root_path+agents_path[index]+'bis_agent_global._2500000.pt', 
            root_path+agents_path[index]+'bis_agent_global._60000.pt',
            root_path+agents_path[index]+'bis_agent_global._800000.pt',
            root_path+agents_path[index]+'bis_agent_global._2180000.pt'
            ]

        
        args[3] = algorithm_name
        
        abl_list = ['_del_graph', '_del_group', '_del_obs']
        del_flag = abl_list[abl_index]     # # abl_flag=0 删除 态势图   # abl_flag=1 删除组别  # abl_flag=2 删除obs
        # del_flag = abl_list[abl_index]   # # abl_flag=0 删除 态势图   # abl_flag=1 删除组别  # abl_flag=2 删除obs
        # del_flag = abl_list[abl_index]   # # abl_flag=0 删除 态势图   # abl_flag=1 删除组别  # abl_flag=2 删除obs
        args.extend(['--abl_flag', del_flag])

        # experiment_name = num_agents + 'agents_' + args[3] + '_ours' + '_num_1'
        # experiment_name = num_agents + 'agents_' + args[3] + '_ours' + '_num_2'
        experiment_name = num_agents + 'agents_' + args[3] + '_ours' + '_num_4' + del_flag
        
        args[5], args[7], args[9] = experiment_name, num_agents, num_agents
        
        # args.extend(['--use_test', 'False', '--use_graph', 'False', '--use_ReLU'])
        args.extend(['--use_test', 'False', '--use_graph', 'True', '--use_ReLU', '--global_bis_agents_path', '', '--local_bis_agents_path', ''])

    
        args[-1] = bis_models_local[index]
        args[-3] = bis_models_global[index]
        
        
        args[13]=str(i)### seed
        main(args)
        ### conda activate pyg; cd /data/jqruan/clustering-162/clustering/on-policy-seeker-164-20230830-abl/onpolicy/scripts/train/2_train/
        ### CUDA_VISIBLE_DEVICES=2 python train_seeker_ours_abl.py 3 
